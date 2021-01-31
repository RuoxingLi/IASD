import sys
import numpy as np
from collections import defaultdict
import json
from ksp import ksp
import math


class NetworkEnv:

    # -------------------------------------- Initialization -----------------------------------------------
    def __init__(self):
        # init load

        self.total_steps = 10
        self.node_map, self.link_spans_dict, self.traffic_matrix, self.distance_matrix = self.state_initialize()
        self.provisioned_matrix = np.array([np.array([0] * len(self.node_map)) for _ in range(len(self.node_map))])
        self.demand_matrix = np.array([np.array([0] * len(self.node_map)) for _ in range(len(self.node_map))])

        self.K = 5  # K shortest paths for candidate
        self.J = 3  # J candidate PP-based DF scheme on each path
        self.ksp = ksp(self.distance_matrix)

        # only considered C-band for now
        self.f_criterion = 193.415  # in THz
        self.w_criterion = 1.55 - 6  # in m
        # self.c = 2.99792458e8  # Speed of light [m/s]

        # note here I only considered 6.25 GHz channel spacing, and FS is never shared with multiple LPs
        self.channel_spacing = 6.25  # in GHz
        self.FS_bandwidth = 12.5  # in GHz
        self.FS_on_link = 350  # Frequency slots per fibre
        # no guard_band considered currently

        # assuming dual-polarization throughout
        self.data_rates = [i for i in range(150, 650, 50)][::-1]
        self.mod_format = [i for i in range(2, 7)][::-1]
        # self.fec_percentage = [i for i in [0, 0.15, 0.27]]
        self.baud_rate_range = [32, 72]

        self.any_guardband = False  # True if we are considering guardbands between lps
        self.guardband_fs = 1  # number of FSs reserved as guardband

        # assuming transparent transmission
        self.noise_figure = 4.3  # EDFA noise figure in db
        self.planks_constant = 6.62607015e-34  # Planck's constant [m^2*kg/s]

        self.global_signal_psd = 26.7e-3  # in [mW/THz]

        self.fs_usage = self.init_link_fs()
        self.fs_usage_temp = self.init_link_fs()
        self.lightpaths = list()  # set of the existing LPs
        self.lightpaths_temp = list()  # set of the existing LPs in temp state
        self.lightpaths_pp = list()  # temp set of LPs used only in Push-Pull based DF

        self.all_configs_list, self.all_configs_dict = self.gen_configs()
        self.center_freq_dict = self.gen_center_frequencies()  # map f_c with FSs of the slice

        self.path_dict = dict()  # for path2link()
        self.path2span_dict = dict()  # for path2span()
        self.span_loss_dict = dict()  # for span_loss()
        self.accumulated_change_dict = dict()  # for accumulated_change()
        self.config_availability_on_path_dict = dict()  # for linear_snr_check()
        self.link_lps_dict = defaultdict(lambda: list())  # for update_link_lps_dict()
        self.link_lps_dict_temp = defaultdict(lambda: list())
        self.path_lps_dict = defaultdict(lambda: list())  # for update_path_lps_dict()
        self.path_lps_dict_temp = defaultdict(lambda: list())

        with open('fiber_parameters.json') as f:
            self.fiber_parameters = json.load(f)
        self.min_snr_dict = self.load_snr_requirement()
        self.acf_a = [0.0, -3.1549, 5.5720, 8.5347e-3, -1.7293, 4.8072e-02,
                      -2.0053e-2, -4.1167e-1, 6.1769e-1, 2.1726e1, 7.9148e-2]  # here the first element is place holder

    @staticmethod
    def state_initialize():
        with open('Links_Germany_17.json') as f:
            link_info = json.load(f)
        with open('Nodes_Germany_17.json') as f:
            node_info = json.load(f)
        with open('Demands_Germany_17_init_traff.json') as f:
            init_traffic_info = json.load(f)

        node_map = defaultdict()
        for k in node_info:
            node = node_info[k]
            node_map[node[0]] = int(k)

        link_spans = defaultdict()
        for k in link_info:
            node_i = node_map[link_info[k]['startNode']]
            node_j = node_map[link_info[k]['endNode']]
            link_spans[node_i * 100 + node_j] = link_info[k]['spanList']
            link_spans[node_j * 100 + node_i] = link_info[k]['spanList'][::-1]

        traffic_matrix = np.array([np.array([0.0] * len(node_map)) for _ in range(len(node_map))])
        for k in init_traffic_info:
            node_i = node_map[init_traffic_info[k][0]]
            node_j = node_map[init_traffic_info[k][1]]
            traffic_matrix[node_i][node_j] = 2 * init_traffic_info[k][2]

        distance_matrix = [[0] * len(node_map) for _ in range(len(node_map))]
        for i in range(len(distance_matrix)):
            distance_matrix[i][i] = 0
        with open('Links_Germany_17.json') as f:
            link_info = json.load(f)
        for link in link_info:
            sp = node_map[link_info[link]['startNode']]
            ep = node_map[link_info[link]['endNode']]
            dis = link_info[link]['linkDist']
            distance_matrix[sp][ep] = math.floor(dis)
            distance_matrix[ep][sp] = math.floor(dis)

        return node_map, link_spans, traffic_matrix, distance_matrix

    def init_link_fs(self):
        fs_usage = dict()
        for node_i in range(len(self.node_map)):
            for node_j in range(len(self.node_map)):
                if self.distance_matrix[node_i][node_j] > 0:
                    # here i assume bidirectional fiber link
                    fs_usage[node_i * 100 + node_j] = [0] * self.FS_on_link
                    fs_usage[node_j * 100 + node_i] = [0] * self.FS_on_link
        return fs_usage

    # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
    def gen_configs(self):
        available_configs = list()
        all_available_configs_dict = defaultdict(list)
        # dual-polarization
        all_configs = [[i, j] for i in self.data_rates for j in self.mod_format]
        for config_i in all_configs:
            baud_rate_i = config_i[0] / (config_i[1] * 2)
            config_i += [baud_rate_i, math.ceil(baud_rate_i / self.FS_bandwidth)]
            if self.baud_rate_range[0] < baud_rate_i < self.baud_rate_range[1]:
                available_configs += [config_i]
        # bitrate: hi -> low, bw_requirement: low -> hi
        available_configs.sort(key=lambda x: (-x[0], x[2]))  # why do I get 74 configs?
        # In HeCSON maybe another filter is minimum SNR - wider in spectrum but higher SNR requirement?
        for config_i in available_configs:
            all_available_configs_dict[config_i[0]] += [config_i]
        return available_configs, all_available_configs_dict

    def gen_center_frequencies(self):
        # generate a dict()
        # key: start_fs, slice_size
        # value: center frequency of this slice (in THz)
        center_freqs_dict = defaultdict()
        slice_size_list = []
        for config in self.all_configs_list:
            if config[3] not in slice_size_list:
                slice_size_list += [config[3]]
        slice_size_list.sort()
        for start_fs in range(self.FS_on_link):
            for slice_size in slice_size_list:
                if not start_fs + slice_size > self.FS_on_link:
                    center_freq = self.f_criterion + (
                            start_fs + 0.5 * slice_size - 0.5 * self.FS_on_link) * self.FS_bandwidth * 1e-3  # in THz
                    center_freqs_dict[str([start_fs, slice_size])] = center_freq
        return center_freqs_dict

    @staticmethod
    def load_snr_requirement():
        min_snr_dict = defaultdict()
        with open('minimum_snr_requirement.json') as f:
            min_snr_data = json.load(f)
        for mod_format in min_snr_data:
            min_snr_dict[min_snr_data[mod_format]['BitsPerSymbol']] = min_snr_data[mod_format]['MinimumSNR']
        return min_snr_dict

    # ------------------------------------ ECOC paper ------------------------------------------------------

    def ecoc_sim(self):
        current_time_step = 0
        while current_time_step <= self.total_steps:
            # get new demand matrix
            self.demand_matrix = self.traffic_matrix - self.provisioned_matrix

            # initialize at t=0
            if current_time_step == 0:
                for node_i, demands_i in enumerate(self.demand_matrix):
                    for node_j, demand_ij in enumerate(demands_i):
                        # RWSA initial demands
                        self.init_rwsa(node_i, node_j, demand_ij)

            else:
                # for demand remaining between i,j
                for node_i, demands_i in enumerate(self.demand_matrix):
                    for node_j, demand_ij in enumerate(demands_i):
                        # demand_remain_ij -> theta_{i,node_j,t}
                        demand_remain_ij = demand_ij

                        # step1: try upgrade existing LPs
                        # get all LPs with s,d = node_i,node_j
                        lightpaths_ij = list()
                        for lp in self.lightpaths:
                            # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
                            path = lp['Path']
                            if path[0] + path[-1] == node_i + node_j and (
                                    path[0] == node_i or path[0] == node_j):
                                lightpaths_ij += [lp]
                        # look for possible upgrade
                        for lp in lightpaths_ij:
                            new_lp, _, _ = self.find_new_config(old_lp=lp)
                            if new_lp:
                                added_capacity = new_lp['DataRate'] - lp['DataRate']
                                self.release_fs(lp['Path'], lp['StartFS'], lp['SliceSize'])
                                self.take_fs(new_lp['Path'], new_lp['StartFS'], new_lp['SliceSize'])
                                self.lightpath_update(lp, new_lp)
                                demand_remain_ij -= added_capacity
                                self.provisioned_matrix[node_i, node_j] += added_capacity
                                if demand_remain_ij <= 0:
                                    break  # all demand serviced

                        # step2: execute ILP to service remaining demands through RWSA
                        if demand_remain_ij > 0:
                            added_capacity = self.ilp_rwsa(node_i, node_j, demand_remain_ij)
                            demand_remain_ij -= added_capacity
                            self.provisioned_matrix[node_i, node_j] += added_capacity

                # step3: check SNR
                is_removed = True
                while is_removed:
                    self.demand_matrix = self.traffic_matrix - self.provisioned_matrix
                    is_removed = self.snr_check()
                    # step3.1: if removed, add new LPs with best-effort using the good old ILP
                    if is_removed:
                        for node_i, demands_i in enumerate(self.demand_matrix):
                            for node_j, demand_remain_ij in enumerate(demands_i):
                                if demand_remain_ij > 0:
                                    added_capacity = self.ilp_rwsa(node_i, node_j, demand_remain_ij)
                                    demand_remain_ij -= added_capacity
                                    self.provisioned_matrix[node_i, node_j] += added_capacity

            # step4: end of a simulated year, save the demand provisioning state
            print('Year', 2020 + current_time_step,
                  '\nDemand:', sum(sum(self.traffic_matrix)),
                  'Provisioned:', sum(sum(self.provisioned_matrix)))
            # then traffic grows
            current_time_step += 1
            self.traffic_matrix *= 1.239  # todo very rough estimation from Fig.2(b) in the ECOC paper
        print('END OF SIMULATION.')

    def find_new_config(self, path=None, old_lp=None, demand=float("Inf"), temp=False):
        # first get channel configs that have bandwidth smaller than or equal to lp
        # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
        # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
        configs = []
        if old_lp:
            for config in self.all_configs_list:
                if config[3] <= old_lp['SliceSize']:
                    configs += [config]
            new_lp, p_ase, p_nli = self.hecson(old_lp['Path'], configs, old_lp, temp)
        elif path:
            # potential bug if datarate changes
            max_50x = max(min(self.data_rates[0] // 50, math.ceil(demand / 50.0)), math.ceil(self.data_rates[-1] // 50))
            for i in range(self.data_rates[-1] // 50, max_50x + 1):
                configs += self.all_configs_dict[i * 50]
            new_lp, p_ase, p_nli = self.hecson(path, configs, temp)
        else:
            print("ERROR find_new_config")
            sys.exit()
        # then run HeCSON to find the best available config
        return new_lp, p_ase, p_nli

    def init_rwsa(self, src, dst, demand):
        demand_remain = demand
        # serve all demand
        while demand_remain > 0:
            # step1: first pick the routing path that has the largest continuous available spectrum slice
            # weighted probabilistic routing based on Yen’s k-Shortest Path Algorithm
            # and the number of continuous empty frequency slots in each of the paths
            paths_seq = self.ksp_weighted_routing(src, dst)

            # step2: then run spectrum assignment
            # use HeCSON to find the best config
            is_found = False
            for path in paths_seq:
                new_lp, _, _ = self.find_new_config(path=path, demand=demand_remain)
                # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
                # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
                if new_lp:
                    is_found = True
                    self.take_fs(new_lp['Path'], new_lp['StartFS'], new_lp['SliceSize'])
                    self.lightpaths += [new_lp]
                    demand_remain -= new_lp['DataRate']
                    self.provisioned_matrix[src, dst] += new_lp['DataRate']
                    break
            if not is_found:
                print("initial RWSA error")
                sys.exit()

    # todo rewrite this part later, current code is not good enough
    def ksp_weighted_routing(self, src, dst):
        paths, _ = self.ksp.k_shortest_paths(src, dst, self.K)
        path_weights = list()
        for path in paths:
            path_usage = self.usage_on_path(path)[::-1]
            if 1 not in path_usage:
                largest_size_i = self.FS_on_link
            else:
                largest_size_i = path_usage.index(1)
            path_weights += [largest_size_i]
        paths_with_weights = sorted([[paths[i], path_weights[i]] for i in range(len(paths))], key=lambda x: x[1])
        sorted_paths = [paths_with_weights[i][0] for i in range(len(paths))]
        return sorted_paths

    def ilp_rwsa(self, src, dst, demand):
        self.update_path_lps_dict()
        added_capacity = 0

        # step1: Find all possible combinations of new channels
        max_provision = max(3, math.ceil(demand / 50.0))
        if demand < 50:
            max_provision = [max_provision * 50]
        else:
            max_provision = [max_provision * 50, (max_provision + 1) * 50]
        lp_combinations = [k for j in [self.get_combinations(i, self.data_rates) for i in max_provision] for k in j]
        lp_combinations = sorted(lp_combinations, key=(lambda x: np.sum(x)))

        # step2: get the sum of the eta_NLI introduced by existing channels between (src,dst)
        # eta_NLI: power independent ACF-NLI co-efficient constraint - devide by power^3
        curr_eta_nli = 0
        candidate_paths, _ = self.ksp.k_shortest_paths(src, dst, self.K)
        for path in candidate_paths:
            lps_src_dst = self.path_lps_dict[str(path)]
            # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
            for lp in lps_src_dst:
                psd_nli_rx = self.acf_egn(lp['Path'], lp['SymbolRate'], lp['CenterFreq'], lp['LaunchPower'])
                curr_eta_nli += psd_nli_rx * lp['SymbolRate'] / (lp['LaunchPower'] ** 3)

        # step3： find and execute one of the possible combination
        # step3.1: classify all the combinations to different ranks based on n_c
        combination_nc_ranks = defaultdict()
        for combination in lp_combinations:
            n_c = np.sum(combination)
            if n_c not in combination_nc_ranks:
                combination_nc_ranks[n_c] = [combination]
            else:
                combination_nc_ranks[n_c] += [combination]

        # step3.2: find a possible combination (first-fit)
        #  save the available scheme(s) if possible
        #  also save the closest scheme if no scheme is available
        available_schemes = list()
        is_any_available = False
        closest_scheme = -1
        closest_provision = -1

        for rank in combination_nc_ranks:
            rank_available = False
            for combination in combination_nc_ranks[rank]:
                # save every found lps and total demand can be provisioned by them
                lps_to_add = []
                p_nli_induced = []
                demand_provisioned = 0
                # note: to make sure all the added lightpaths do not overlap and have enough SNR,
                #       I have used 'temp' network state.
                self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths)
                self.fs_usage_temp = self.fs_usage_deepcopy(self.fs_usage)

                # Then calculate the availability and the newly-introduced eta_NLI of the current scheme
                for dr_index, channels_to_add in enumerate(combination):
                    # probabilistic_routing + HeCSON to check the availability of the current combination
                    configs = self.all_configs_dict[self.data_rates[dr_index]]
                    for i in range(channels_to_add):
                        paths_seq = self.ksp_weighted_routing(src, dst)
                        # use HeCSON to find the best config
                        for path in paths_seq:
                            new_lp, p_ase, p_nli = self.hecson(path, configs, temp=True)
                            # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
                            if new_lp:
                                # if found available config, save it and update the 'temp' spectrum state
                                self.take_fs(new_lp['Path'], new_lp['StartFS'], new_lp['SliceSize'], temp=True)
                                self.lightpaths_temp += [new_lp]
                                lps_to_add += [new_lp]
                                p_nli_induced += [p_nli]
                                demand_provisioned += new_lp['DataRate']
                                break
                if demand_provisioned >= demand:  # if current combination is available
                    #  check NLI - create a LIST
                    #  if the added noise is too much, only save the eta-eligible ones for closest
                    added_eta_nli = [0]
                    # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
                    for i, lp in enumerate(lps_to_add):
                        added_eta_nli += [added_eta_nli[-1] + p_nli_induced[i] / (lp['LaunchPower'] ** 3)]
                    added_eta_nli = added_eta_nli[1:]
                    if added_eta_nli[-1] <= curr_eta_nli:  # if satisfied constraint (2)
                        rank_available = True
                        available_schemes += [lps_to_add]
                        continue
                    else:  # if not satisfied, strictly limit the NLI.
                        for i in range(len(lps_to_add)):
                            if added_eta_nli[i] > curr_eta_nli:
                                lps_to_add = lps_to_add[:i]
                                demand_provisioned = np.sum([lp['DataRate'] for lp in lps_to_add])
                                break
                if demand_provisioned > closest_provision:
                    closest_scheme = lps_to_add
                    closest_provision = demand_provisioned
            if rank_available:
                is_any_available = True
                # randomly pick and execute one of the available combinations in this rank
                lps_to_add = available_schemes[np.random.randint(len(available_schemes))]
                for lp in lps_to_add:
                    self.take_fs(lp['Path'], lp['StartFS'], lp['SliceSize'])
                    self.lightpaths += [lp]
                    added_capacity += lp['DataRate']
                break
        if not is_any_available:
            # if there is no available combination of lps that satisfy the constraints,
            # pick the one that has max added capacity satisfying constraint (2).
            lps_to_add = closest_scheme
            for lp in lps_to_add:
                self.take_fs(lp['Path'], lp['StartFS'], lp['SliceSize'])
                self.lightpaths += [lp]
            added_capacity += closest_provision
        return added_capacity

    @staticmethod
    def get_combinations(demand, data_rates):
        tx_combinations = list()
        remaining_demand_list = list()
        remaining_demand_list += [demand]
        for i, data_rate_i in enumerate(data_rates):
            new_tx_combination = list()
            new_remaining_demand_list = list()
            for j, remaining_demand in enumerate(remaining_demand_list):
                if remaining_demand > 0:
                    if remaining_demand % data_rate_i == 0:
                        max_tx_i = remaining_demand // data_rate_i
                    else:
                        max_tx_i = remaining_demand // data_rate_i + 1
                else:
                    max_tx_i = 0
                tx_list_i = [k for k in range(max_tx_i + 1)]
                if i == 0:
                    new_tx_combination = [[k] for k in tx_list_i]
                    new_remaining_demand_list += [remaining_demand - data_rate_i * k for k in tx_list_i]
                elif i == len(data_rates) - 1:
                    tx_list_i = [max_tx_i]
                    temp = [tx_combinations[j] + [tx_list_i[k]] for k in range(len(tx_list_i))]
                    new_tx_combination += temp
                    new_remaining_demand_list += [remaining_demand - data_rate_i * k for k in tx_list_i]
                else:
                    temp = [tx_combinations[j] + [tx_list_i[k]] for k in range(len(tx_list_i))]
                    new_remaining_demand_list += [remaining_demand - data_rate_i * k for k in tx_list_i]
                    new_tx_combination += temp
            tx_combinations = new_tx_combination
            remaining_demand_list = new_remaining_demand_list
        tx_combinations_temp = tx_combinations[:]
        remaining_demand_list_temp = remaining_demand_list[:]
        for i, combination in enumerate(tx_combinations_temp):
            if remaining_demand_list_temp[i] < 0:
                tx_combinations.remove(combination)
                remaining_demand_list.remove(remaining_demand_list_temp[i])
        return tx_combinations

    # -------------------------------- OFC paper: HeCSON --------------------------------------------------

    # note here if we use 'temp' spectrum state, remember to update self.lightpaths_temp
    def hecson(self, path, configs, origin_lp=None, temp=False):
        config_found = False
        new_lp = None
        p_ase = None
        p_nli = None
        if origin_lp:
            # if upgrading an existing LP, keep f_cut unchanged
            for config in configs:
                # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
                # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
                config_found, p_ase, p_nli = self.self_check(path, config[1], config[2],
                                                             origin_lp['CenterFreq'], temp=temp)
                if config_found:
                    # note here the number of used FS correlates to center frequency,
                    #   and thus may increase due to the position of center frequency.
                    f_pos = origin_lp['StartFS'] + origin_lp['SliceSize'] * 0.5
                    fs_each_side = config[3] * 0.5
                    start_fs = math.floor(f_pos - fs_each_side)
                    slice_size = int((f_pos - start_fs) * 2)
                    new_lp = self.gen_lightpath(path, config[0], config[1], config[2], start_fs, slice_size,
                                                origin_lp['CenterFreq'], origin_lp['LaunchPower'], p_ase, p_nli)
                    break
        else:
            # if setting up a new LP, select f_cut based on the first-fit spectrum assignment scheme
            for config in configs:
                # for every config, find the first possible slice (by first-fit)
                slice_found = self.find_slice(path, config[3], slice2find=1, temp=temp)
                # then validate config
                if slice_found:
                    first_slice_index = slice_found[0]
                    f_center_i = self.center_freq_dict[str([first_slice_index, config[3]])]
                    config_found, p_ase, p_nli = self.self_check(path, config[1], config[2], f_center_i,
                                                                 temp=temp)
                    if config_found:
                        p_ch = self.assign_channel_power(config[2])
                        new_lp = self.gen_lightpath(path, config[0], config[1], config[2], first_slice_index,
                                                    config[3], f_center_i, p_ch, p_ase, p_nli)
                        break
                if config_found:
                    break
        return new_lp, p_ase, p_nli

    # --------------------------------------- SNR -------------------------------------------------------
    def snr_check(self):
        flag = False  # True if all pass, false if the SNR of any channel failed to reach the threshold
        temp_lp_list = self.lightpaths[:]
        self.update_link_lps_dict()
        for lp in temp_lp_list:
            # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
            check_pass, p_ase, p_nli = self.self_check(lp['Path'], lp['ModFormat'], lp['SymbolRate'],
                                                       lp['CenterFreq'], lp['LaunchPower'], need_update=False)
            lp['ASE'] = p_ase
            lp['NLI'] = p_nli
            if not check_pass:
                flag = True
                self.drop_lp(lp, temp=False)
                # update self.provisioned_matrix
                self.provisioned_matrix[lp['Path'][0], lp['Path'][-1]] -= lp['DataRate']
        return flag

    # here I used SNR instead of OSNR that Sai used in ECOC & OFC
    # note here if we use 'temp' spectrum state, remember to update self.lightpaths_temp
    def self_check(self, path, mod_format, r_cut, f_cut, p_cut=None, filter_extended=None,
                   need_update=True, temp=False):
        if need_update:
            self.update_link_lps_dict(temp)
        if p_cut is None:
            p_cut = self.assign_channel_power(r_cut)
        minimum_snr = self.min_snr_dict[mod_format]

        # linear SNR
        p_ase_cut = self.ase_noise_power(path, r_cut, f_cut, filter_extended)
        snr_linear = p_cut / p_ase_cut
        if snr_linear < minimum_snr:
            return False

        # NLI SNR
        psd_nli_cut = self.acf_egn(path, r_cut, f_cut, p_cut, temp)
        p_nli_cut = psd_nli_cut * r_cut
        snr_acf = p_cut / (p_ase_cut + p_nli_cut)
        if snr_acf < minimum_snr:
            psd_nli_cut = self.ff_egn(path, r_cut, f_cut, temp)
            p_nli_cut = psd_nli_cut * r_cut
            snr_ff = p_cut / (p_ase_cut + p_nli_cut)
            if snr_ff < minimum_snr:
                return False
        return True, p_ase_cut, p_nli_cut

    # span loss in db
    def span_loss(self, span, returning='Linear'):
        if str(span) in self.span_loss_dict:
            span_loss = self.span_loss_dict[str(span)]
        else:
            span_loss = span['SpanLength'] * span['attnDB']
            self.span_loss_dict[str(span)] = span_loss
        if returning == 'Linear':
            return span_loss
        elif returning == 'dB':
            return 10 * math.log10(span_loss)

    def linear_snr_check(self, path, config):
        key = str([path, config])
        if key in self.config_availability_on_path_dict:
            return self.config_availability_on_path_dict[key]
        else:
            p_ch = self.assign_channel_power(config[2])
            # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
            p_ase = self.ase_noise_power(path, config[2], self.center_freq_dict[str([0, config[3]])])
            linear_snr_db = 10 * math.log10(p_ch / p_ase)
            check_result = linear_snr_db < self.min_snr_dict[config[1]]
            self.config_availability_on_path_dict[key] = check_result
            return check_result

    # ---------------------------------- ASE Noise ---------------------------------------------------------------

    def ase_noise_power(self, path, r_cut, f_cut, filter_width=None):
        pdf_ase = self.ase_noise_psd(path, f_cut)
        if filter_width:
            p_ase_cut_with_filter = pdf_ase * filter_width
            return p_ase_cut_with_filter
        else:
            p_ase_cut = pdf_ase * r_cut
            return p_ase_cut

    def ase_noise_psd(self, path, f_cut):
        # much help from Sai
        pdf_ase = 0
        spans_on_path = self.path2span(path)
        for span in spans_on_path:
            linear_noise_figure = 10 ** (self.noise_figure / 10)
            linear_span_loss = self.span_loss(span, returning='Linear')
            # Kumar Book [6.104]
            spont_emission = linear_noise_figure / 2 * linear_span_loss / (linear_span_loss - 1) - 1 / linear_span_loss
            # [7.15] Becker EDFA Book
            pdf_ase += spont_emission * self.planks_constant * f_cut * (linear_span_loss - 1)
        return pdf_ase

    # ---------------------------------- ACF-EGN -----------------------------------------------------------------
    # todo not tested yet

    def acf_egn(self, path, r_cut, f_cut, p_cut, temp=False):
        # NOTE: Frequencies and bandwidths are in THz and PSDs are in W/THz.
        psd_nli_rx = 0
        r_cut *= 1e-3  # symbol rate in THz
        spans_on_path = self.path2span(path)
        for span_index, span in enumerate(spans_on_path):
            psd_nli_rx += self.acf_psd_nli_n(path, span, r_cut, f_cut, p_cut, temp) * self.apc(path, span_index)
        return psd_nli_rx

    def acf_psd_nli_n(self, path, curr_span, r_cut, f_cut, p_cut, temp=False):
        psd_nli_n = (16 / 27) * (self.fnlc(curr_span['FiberType']) ** 2) * p_cut / r_cut

        # self channel interference - SCI
        sci = self.acf_sci(path, curr_span, f_cut, r_cut, p_cut)

        # cross channel interference - XCI
        xci = 0
        # first get all lightpaths (channels) except the CUT lightpath.
        links_on_path = self.path2link(path)
        curr_link = -1
        for link in links_on_path:
            spans_on_link = self.path2span(link)
            if curr_span in spans_on_link:
                curr_link = link
                break
        if temp:
            lps_on_span = self.link_lps_dict_temp[curr_link[0] * 100 + curr_link[1]]
        else:
            lps_on_span = self.link_lps_dict[curr_link[0] * 100 + curr_link[1]]
        # lp_i: [path, data_rate, mod_format, fec, symbol_rate, start_fs, slice_size, center_freq, p_ch]
        for lp_i in lps_on_span:
            if lp_i['CenterFreq'] == f_cut:
                continue
            xci += self.afc_xci(lp_i, curr_span, f_cut, r_cut)
        psd_nli_n *= sci + xci
        return psd_nli_n

    def acf_sci(self, path, curr_span, f_cut, r_cut, p_cut):
        # rho_cut * psd(cut) ** 2 * I_cut
        rho_cut = self.acf_a[1] + self.acf_a[2] * r_cut ** self.acf_a[3] + self.acf_a[4] * (
                math.fabs(self.beta_acc(path, curr_span, f_cut, f_cut)) + self.acf_a[5]) ** self.acf_a[6]
        psd_cut = p_cut / r_cut
        i_cut = self.acf_i(curr_span, f_cut, f_cut, r_cut, r_cut)
        return rho_cut * psd_cut ** 2 * i_cut

    def afc_xci(self, lp, curr_span, f_cut, r_cut):
        # Note: for each channel (lightpath) n_ch != n_cut: 2 * rho_nch * psd(nch) ** 2 * i_nch
        rho_nch = self.acf_a[7] + self.acf_a[8] * (math.fabs(
            self.beta_acc(lp['Path'], curr_span, f_cut, lp['CenterFreq'])) + self.acf_a[9]) ** self.acf_a[10]
        psd_nch = lp['LaunchPower'] / (lp['SymbolRate'] * 1e-3)
        i_nch = self.acf_i(curr_span, f_cut, lp['CenterFreq'], r_cut, lp['SymbolRate'] * 1e-3)
        xci_nch = 2 * rho_nch * psd_nch ** 2 * i_nch
        return xci_nch

    def acf_i(self, span, f_cut, f_nch, r_cut, r_ch):  # in THz, TBaud
        fraction = math.asinh(math.pi ** 2 * math.fabs(self.beta_2_nch(span, f_cut, f_nch) / (
                2 * self.attenuation(span['FiberType']))) * (f_nch - f_cut + 0.5 * r_ch) * r_cut) - \
                   math.asinh(math.pi ** 2 * math.fabs(self.beta_2_nch(span, f_cut, f_nch) / (
                           2 * self.attenuation(span['FiberType']))) * (f_nch - f_cut - 0.5 * r_ch) * r_cut)
        denominator = 4 * math.pi * math.fabs(self.beta_2_nch(span, f_cut, f_nch)) * 2 * self.attenuation(
            span['FiberType'])
        return fraction / denominator

    def beta_acc(self, path, curr_span, f_cut, f_ch):
        spans_on_path = self.path2span(path)
        beta_acc = 0
        for span in spans_on_path:
            if span == curr_span:
                break
            beta_acc += self.beta_2_nch(span, f_cut, f_ch) * span['SpanLength']
        return beta_acc

    def beta_2_nch(self, span, f_cut, f_nch):
        beta_2 = self.dispersion(span['FiberType'])
        beta_3 = self.dispersion_slope(span['FiberType'])
        return beta_2 + math.pi * beta_3 * (f_cut + f_nch - 2 * self.f_criterion)

    def apc(self, path, span_index):  # Accumulative Power Change
        if str(path) in self.accumulated_change_dict:
            return self.accumulated_change_dict[str(path)][span_index]
        else:
            spans_on_path = self.path2span(path)[::-1]
            accumulated_power_change = list()
            current_accumulated_power_change = 1
            for span in spans_on_path:
                current_accumulated_power_change *= span['EDFAGain'] * (
                        math.e ** (-2 * self.span_loss(span)))
                accumulated_power_change += [current_accumulated_power_change]
            self.accumulated_change_dict[str(path)] = accumulated_power_change[::-1]
            return self.accumulated_change_dict[str(path)][span_index]

    def cross_snr_check_and_update(self, path, target_lp, direction=None, add_or_drop='Add', temp=False, update=True):
        links_using = self.path2link(path)
        lps_checked = list()
        nli2change = list()
        lp_list_used = self.lightpaths_temp if temp else self.lightpaths

        if add_or_drop != 'Add' and add_or_drop != 'Drop':
            print("error cross check")
            sys.exit()

        # todo the efficiency of these lines can be improved greatly, if have time later
        for link in links_using:
            lps_to_check = self.filter_lps_on_link(link, target_lp['StartFS'], direction=direction, temp=temp)
            for lp_i in lps_to_check:
                if lp_i not in lps_checked:
                    shared_span_list = self.shared_span_cut(lp_i['Path'], target_lp['Path'])
                    changed_p_nli = 0
                    for span_shared in shared_span_list:
                        changed_psd_nli = self.afc_xci(target_lp, span_shared[0], lp_i['CenterFreq'],
                                                       lp_i['SymbolRate'])
                        changed_psd_nli *= (16 / 27) * (self.fnlc(span_shared[0]['FiberType']) ** 2
                                                        ) * (lp_i['LaunchPower'] / lp_i['SymbolRate'] * 1e-3)
                        changed_psd_nli *= self.apc(lp_i['Path'], span_shared[1])
                        changed_p_nli += changed_psd_nli * lp_i['SymbolRate'] * 1e-3
                    if add_or_drop == 'Add':
                        snr_i = lp_i['LaunchPower'] / (lp_i['ASE'] + lp_i['NLI'])
                        snr_min_i = self.min_snr_dict[lp_i['ModFormat']]
                        lps_checked += [lp_i]
                        nli2change += [changed_p_nli]
                        # Case A: If the snr of lp_i has already dropped below the threshold,
                        #         the impact of df on lp_i is ignored
                        if snr_i < snr_min_i:
                            continue
                        # Case B: If the snr of lp_i is above the threshold before df,
                        #         make sure it won't drop below the threshold!
                        _snr_i = lp_i['LaunchPower'] / (lp_i['ASE'] + lp_i['NLI'] + changed_p_nli)
                        if _snr_i < snr_min_i:
                            return False
                    else:
                        changed_p_nli = - changed_p_nli
                        lps_checked += [lp_i]
                        nli2change += [changed_p_nli]

        # if all pass, update if needed!
        if update:
            for i, lp in enumerate(lps_checked):
                lp_list_used[lp_list_used.index(lp)]['NLI'] += nli2change[i]
        return True

    # ------------------------------- Full-form EGN ------------------------------------------------------------

    # todo not written yet, kind of challenging to implement
    def ff_egn(self, path, r_cut, f_cut, temp=False):
        if temp == 'check':
            print(self.FS_on_link, r_cut, f_cut, temp, path)
        return float('Inf')

    # -------------------- Reactive Make-before-Break Defragmentation (Cost-First) -------------------------------------

    # check config
    def ia_mbb_re(self, src, dst, demand, chosen_path=None):
        available_schemes = []
        # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
        config_ranks = defaultdict(list)
        max_provision = min(max(self.data_rates[0] // 50, math.ceil(demand / 50.0)), self.data_rates[-1] // 50)
        for i in range(max_provision, self.data_rates[0] // 50 - 1, -1):
            config_ranks[i * 50] += self.all_configs_dict[i * 50]

        if chosen_path:
            candidate_paths = [chosen_path]
        else:
            candidate_paths = self.ksp_weighted_routing(src, dst)

        possible_df_scheme = list()
        for rank in config_ranks:
            for config in config_ranks[rank]:
                for path in candidate_paths:
                    self.update_link_lps_dict()  # get current linkUsageDict
                    links_on_path = self.path2link(path)
                    lps_on_path = []
                    for link in links_on_path:
                        lps_on_path += self.link_lps_dict[link[0] * 100 + link[1]]
                    lps_on_path = [lp for x, lp in enumerate(lps_on_path) if lps_on_path.index(lp) == x]
                    if not self.linear_snr_check(path, config):
                        continue
                    # step1: Start searching from FS indexed 0
                    for slice_start in range(self.FS_on_link - config[3]):
                        # step1.1: search for lps on path and conflict to the interval
                        slice_end = slice_start + config[3] - 1
                        conflict_lps = []
                        for lp in lps_on_path:
                            # step1.2: get all the conflict lps on the path, save it in lp_conflict
                            cond_1 = slice_start <= lp['StartFS'] <= slice_end
                            cond_2 = slice_start <= lp['StartFS'] + lp['SliceSize'] - 1 <= slice_end
                            cond_3 = lp['StartFS'] < slice_start and slice_end < lp['StartFS'] + lp['SliceSize'] - 1
                            if cond_1 or cond_2 or cond_3:
                                conflict_lps.append(lp)

                        # 这里可以先初步筛查一下，可以减少一点点后面的工作量
                        # 具体实现是，把其他相互冲突的光路先撤除了，然后把目标光路先建立起来，看snr够不够
                        self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths)
                        self.fs_usage_temp = self.fs_usage_deepcopy(self.fs_usage)
                        for lp in conflict_lps:
                            self.drop_lp(lp, temp=True)
                        f_c = self.center_freq_dict[str([slice_start, config[3]])]
                        p_ch = self.assign_channel_power(config[2])
                        target_lp = self.gen_lightpath(path, config[0], config[1], config[2], slice_start, config[3],
                                                       f_c, p_ch)
                        is_added, target_lp = self.add_lp(target_lp, temp=True)
                        if is_added:
                            possible_df_scheme += [[path, slice_start, conflict_lps, target_lp, config[3]]]

        # Step2: rank the possible spectrum positions by:
        #           1) capacity of the new channel
        #           2) number of lps to be reconfigured
        scheme_ranks = defaultdict(list)
        for scheme in possible_df_scheme:
            scheme_ranks[scheme[3]['DataRate']] += [scheme]

        # Step3: search for possible df schemes rank by rank
        for curr_rank in scheme_ranks:
            rank_available = False
            least_reconfiguration = float('Inf')
            scheme_ranks[curr_rank] = sorted(scheme_ranks[curr_rank], key=lambda x: x[2])
            for scheme in scheme_ranks[curr_rank]:
                if scheme[2] > least_reconfiguration:
                    break
                conflict_lp_new = list()
                self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths)
                self.fs_usage_temp = self.fs_usage_deepcopy(self.fs_usage)
                for lp in scheme[2]:
                    self.update_link_lps_dict(temp=True)
                    lp_reconfigured = False
                    paths = self.ksp.k_shortest_paths(lp['Path'][0], lp['Path'][-1], self.K)
                    for path in paths:
                        # Step3.1: look for possible free spectrum to reconfigure conflicting lps (first-fit)
                        for config in self.all_configs_dict[lp['DataRate']]:
                            if not self.linear_snr_check(path, config):
                                continue
                            # for every config, find the possible slices
                            slice_indexes = self.find_slice(path, config[3], temp=True)
                            for slice_i in slice_indexes:
                                if path == scheme[0]:
                                    if not slice_i + scheme[4] <= scheme[1] and not slice_i >= scheme[1] + scheme[4]:
                                        continue
                                f_c = self.center_freq_dict[str([slice_i, config[3]])]
                                p_ch = self.assign_channel_power(config[2])
                                reconfigured_lp = self.gen_lightpath(path, config[0], config[1], config[2],
                                                                     slice_i, config[3], f_c, p_ch)
                                lp_reconfigured, reconfigured_lp = self.add_lp(reconfigured_lp, temp=True)
                                if not lp_reconfigured:
                                    continue
                                self.drop_lp(lp, temp=True)
                                conflict_lp_new += [reconfigured_lp]
                                break
                            if lp_reconfigured:
                                break  # if pass break it
                        if lp_reconfigured:
                            break  # if pass break it
                    if not lp_reconfigured:
                        break
                if len(conflict_lp_new) == len(scheme[2]):
                    # check the availability of the target lp
                    target_lp = scheme[3]
                    target_lp_added, target_lp = self.add_lp(target_lp, temp=True)
                    if target_lp_added:
                        # save it
                        available_schemes += [[self.lightpaths_temp, self.fs_usage_temp]]
                        rank_available = True
                        least_reconfiguration = min(len(conflict_lp_new), least_reconfiguration)
            if rank_available:
                # break it
                break
        if available_schemes:
            # there might be better options, will work on it later
            scheme_chosen = available_schemes[np.random.randint(len(available_schemes))]
            # apply the chosen df scheme
            self.lightpaths = scheme_chosen[0]
            self.fs_usage_temp = scheme_chosen[1]
            mbb_cost = scheme_chosen[2]
            new_lp = scheme_chosen[3]
            return True, mbb_cost, new_lp
        else:
            return False, None, None

    # ---------------------------- Reactive Push-Pull based cost-first DF --------------------------------------------

    # basically based on the Rui Wang's defragmentation proposed in the JLT paper
    # a remaining challenge is that current code does not support input state with low_SNR channels
    def ia_pp_re(self, src, dst, demand, chosen_path=None):
        # reactive defrag, wang's optimal defrag is executed while keeping action = 0
        reconfig_available = False
        available_schemes = list()

        # Step1: get the deltas, ia_deltas for ALL the existing OPs in the EON
        # lightpaths_pp saves original lightpath list
        # while lightpaths_temp is changing with with this 保存原始光路信息与ex，all_ops则随着defrag变动
        self.lightpaths_pp = self.lightpath_list_deepcopy(self.lightpaths)
        self.lightpaths_pp = sorted(self.lightpaths, key=(lambda x: x['StartFS'] + x['SliceSize'] - 1))

        self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths_pp)
        self.fs_usage_temp = self.fs_usage_deepcopy(self.fs_usage)

        # Step1.1: perform delta_defrag_left for all the existing ops, save e(x) for each lp
        for i, lp in enumerate(self.lightpaths_temp):
            # NOTE here spectral_floor_i * 2 + 1 is fixed for only 12.5/6.25 GHz FS bandwidth and channel spacing
            #      therefore bugs may exist when changing these to other values

            # get the e_delta(x) for lp in lightpaths_temp
            original_center_freq = lp['CenterFreq']
            max_spacing_shift_lower = self.delta_defrag(lp, "Left")
            # find out the largest distance lp can be shifted when considering physical layer
            lowest_center_freq_available, lp = self.pp_shift_check(lp, max_spacing_shift_lower, direction='Left')

            center_fs_pos = self.freq_2_fs(lowest_center_freq_available)
            fs_each_side = lp['SymbolRate'] / (2 * lp['ModFormat']) * 0.5
            start_fs = math.floor(center_fs_pos - fs_each_side)
            slice_size = int((center_fs_pos - start_fs) * 2)

            self.lightpaths_pp[i]['Original_StartFS'] = self.lightpaths_pp[i]['StartFS']
            self.lightpaths_pp[i]['Original_SliceSize'] = self.lightpaths_pp[i]['SliceSize']
            self.lightpaths_pp[i]['Original_CenterFreq'] = self.lightpaths_pp[i]['CenterFreq']

            self.lightpaths_pp[i]['Lowest_StartFS'] = start_fs
            self.lightpaths_pp[i]['Lowest_SliceSize'] = slice_size
            self.lightpaths_pp[i]['Lowest_CenterFreq'] = lowest_center_freq_available
            self.lightpaths_pp[i]['Lowest_SliceEnd'] = start_fs + slice_size - 1

            # if shifted successfully, update fs_usage
            if lowest_center_freq_available != original_center_freq:
                # SNR of the channels are already updated in function pp_shift_check
                self.lightpaths_temp[i]['StartFS'] = start_fs
                self.lightpaths_temp[i]['SliceSize'] = slice_size
                self.lightpaths_temp[i]['CenterFreq'] = lowest_center_freq_available
                self.release_fs(self.lightpaths_pp[i]['Path'], self.lightpaths_pp[i]['StartFS'],
                                self.lightpaths_pp[i]['SliceSize'], temp=True)
                self.take_fs(self.lightpaths_temp[i]['Path'], self.lightpaths_temp[i]['StartFS'],
                             self.lightpaths_temp[i]['SliceSize'], temp=True)

        # Step1.2: perform reverse_delta_defrag for all the existing ops, save s(x) for each lp
        self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths_pp)
        self.fs_usage_temp = self.fs_usage_deepcopy(self.fs_usage)
        for i, lp in enumerate(reversed(self.lightpaths_temp)):
            # get the e_delta(x) for lp in lightpaths_temp
            original_center_freq = lp['CenterFreq']
            max_spacing_shift_higher = self.delta_defrag(lp, "Right")  # in numbers of minimum spacing gaps
            # find out the largest distance lp can be shifted when considering physical layer
            highest_center_freq_available, lp = self.pp_shift_check(lp, max_spacing_shift_higher, direction='Right')

            center_fs_pos = self.freq_2_fs(highest_center_freq_available)
            fs_each_side = lp['SymbolRate'] / (2 * lp['ModFormat']) * 0.5
            start_fs = math.floor(center_fs_pos - fs_each_side)
            slice_size = int((center_fs_pos - start_fs) * 2)

            j = len(self.lightpaths_temp) - i - 1
            self.lightpaths_pp[j]['Highest_StartFS'] = start_fs
            self.lightpaths_pp[j]['Highest_SliceSize'] = slice_size
            self.lightpaths_pp[j]['Highest_CenterFreq'] = highest_center_freq_available

            # if shifted successfully, update fs_usage
            if highest_center_freq_available != original_center_freq:
                self.lightpaths_temp[j]['StartFS'] = start_fs
                self.lightpaths_temp[j]['SliceSize'] = slice_size
                self.lightpaths_temp[j]['CenterFreq'] = highest_center_freq_available
                self.release_fs(self.lightpaths_pp[j]['Path'], self.lightpaths_pp[j]['StartFS'],
                                self.lightpaths_pp[j]['SliceSize'], temp=True)
                self.take_fs(self.lightpaths_temp[j]['Path'], self.lightpaths_temp[j]['StartFS'],
                             self.lightpaths_temp[j]['SliceSize'], temp=True)

        # Step2.0: for given path, sort CS(p) in ascending order of e_delta_x
        # self.lightpaths_pp = sorted(self.lightpaths_pp, key=(lambda x: x['Lowest_StartFS']))  #  not a must
        self.lightpaths_temp = self.lightpath_list_deepcopy(self.lightpaths_pp)
        self.update_link_lps_dict(temp=True)

        # Step2.1: calculate the potential slice size for each interval on the candidate paths, and save them
        if chosen_path:
            candidate_paths = [chosen_path]
        else:
            candidate_paths = self.ksp_weighted_routing(src, dst)

        all_possible_intervals = list()
        # first find all the intervals on each candidate path
        for path in candidate_paths:
            lps_on_path = list()
            links_on_path = self.path2link(path)
            for link in links_on_path:
                lps_on_path += self.link_lps_dict[link[0] * 100 + link[1]]
            lps_on_path = [lp for x, lp in enumerate(lps_on_path) if lps_on_path.index(lp) == x]  # remove redundancy
            lps_on_path = sorted(lps_on_path, key=(lambda x: x['Lowest_SliceEnd']))
            # find all the intervals, and get their largest possible size considering physical layer impairments
            intervals_on_path = list(set([0] + [lp['Lowest_SliceEnd'] for lp in lps_on_path]))
            for interval_position in intervals_on_path:
                lps_lower = list()
                lps_higher = list()
                for lp in lps_on_path:
                    if lp['Lowest_SliceEnd'] <= interval_position:
                        lps_lower += [lp]
                    else:
                        lps_higher += [lp]

                # calculate f_before, f_after, c_before, c_after, b_before, b_max
                if lps_lower:
                    f_before = max([x['StartFS'] + x['SliceSize'] - 1 for x in lps_lower])  # max e(x) in op_lower
                    f_after = max([x['Lowest_SliceEnd'] for x in lps_lower])  # max e_delta(x) in op_lower
                else:
                    f_before = -1
                    f_after = -1
                if lps_higher:
                    c_before = min([x['StartFS'] for x in lps_higher])  # min s(x) in op_higher
                    c_after = min([x['Highest_StartFS'] for x in lps_higher])  # min s_rev_delta(x) in op_higher
                else:
                    c_before = self.FS_on_link
                    c_after = self.FS_on_link

                b_before = c_before - f_before - 1  # size of interval - before
                b_max = c_after - f_after - 1  # size of interval - maximum
                max_delta_low = f_before - f_after
                max_delta_high = c_after - c_before
                interval_original_center_freq = self.interval_original_center_freq(f_before, c_before, b_before)

                left_gb = False
                right_gb = False
                if self.any_guardband:
                    # see if needed to reserve guard_band based on the indexes of the interval
                    # here we mark where the guard band is reserved
                    if f_after != -1:
                        left_gb = True
                    if c_after != self.FS_on_link:
                        right_gb = True
                b_available = b_max - (left_gb + right_gb) * self.guardband_fs
                # a possible interval has to be at least larger to accommodate the smallest channel
                if b_available < math.ceil(self.baud_rate_range[0] / self.FS_bandwidth):
                    continue

                # For intervals that can be extended to accommodate lightpaths:

                # 1. Get all the lightpaths that may need to be reconfigured
                # 1.1 Get all the lightpaths that may need to be shifted to lower spectrum
                relevant_lps_lower_temp = lps_lower[:]  # note here is simple copy not deepcopy
                relevant_lps_lower = list()
                link_scanned = sum([[x, x[::-1]] for x in self.path2link(path)], [])
                while relevant_lps_lower_temp:
                    relevant_lps_lower_temp = sorted(relevant_lps_lower_temp, key=lambda x: -x['StartFS'])
                    lp = relevant_lps_lower_temp.pop(0)
                    if lp in relevant_lps_lower:
                        continue
                    links_lp_traversed = self.path2link(lp['Path'])
                    for link in links_lp_traversed:
                        if link not in link_scanned:
                            found_lps = self.filter_lps_on_link(link, lp['StartFS'], direction='Left', temp=True)
                            # relevant_lps_lower_temp += [lp for lp in found_lps if lp not in relevant_lps_lower_temp]
                            relevant_lps_lower_temp += found_lps
                            relevant_lps_lower_temp = [lp for x, lp in enumerate(relevant_lps_lower_temp) if
                                                       relevant_lps_lower_temp.index(lp) == x]
                            link_scanned += [link, link[::-1]]
                    relevant_lps_lower += [lp]

                # 1.2 Get all the lightpaths that may need to be shifted to higher spectrum
                relevant_lps_higher_temp = lps_higher[:]  # note here is simple copy not deepcopy
                relevant_lps_higher = list()
                link_scanned = sum([[x, x[::-1]] for x in self.path2link(path)], [])
                while relevant_lps_higher_temp:
                    relevant_lps_higher_temp = sorted(relevant_lps_higher_temp,
                                                      key=lambda x: x['StartFS'] + x['SliceSize'] - 1)
                    lp = relevant_lps_higher_temp.pop(0)
                    if lp in relevant_lps_higher:
                        continue
                    links_lp_traversed = self.path2link(lp['Path'])
                    for link in links_lp_traversed:
                        if link not in link_scanned:
                            found_lps = self.filter_lps_on_link(link, lp['StartFS'], direction='Right', temp=True)
                            # relevant_lps_higher_temp += [lp for lp in found_lps if lp not in relevant_lps_higher_temp]
                            relevant_lps_higher_temp += found_lps
                            relevant_lps_higher_temp = [lp for x, lp in enumerate(relevant_lps_higher_temp) if
                                                        relevant_lps_higher_temp.index(lp) == x]
                            link_scanned += [link, link[::-1]]
                    relevant_lps_higher += [lp]

                # 1.3 Get all the lightpaths that need not be reconfigured for this interval
                # fixed_lps = [lp for lp in self.lightpaths_temp if
                #              lp not in relevant_lps_lower and lp not in relevant_lps_higher]

                # 2. Reconfigure all the relevant lps to its lowest/highest positions, update SNR of relevant lps.
                for lp in relevant_lps_lower + relevant_lps_higher:
                    self.drop_lp(lp, temp=True)
                for lp in relevant_lps_lower:
                    lp['StartFS'] = lp['Lowest_StartFS']
                    lp['SliceSize'] = lp['Lowest_SliceSize']
                    lp['CenterFreq'] = lp['Lowest_CenterFreq']
                    is_added, reconfigured_lp = self.add_lp(lp, temp=True)
                    lp['ASE'] = reconfigured_lp['ASE']
                    lp['PLI'] = reconfigured_lp['PLI']
                    if not is_added:
                        print('Logic Wrong in L1085')
                        sys.exit()
                for lp in relevant_lps_higher:
                    lp['StartFS'] = lp['Highest_StartFS']
                    lp['SliceSize'] = lp['Highest_SliceSize']
                    lp['CenterFreq'] = lp['Highest_CenterFreq']
                    is_added, reconfigured_lp = self.add_lp(lp, temp=True)
                    lp['ASE'] = reconfigured_lp['ASE']
                    lp['PLI'] = reconfigured_lp['PLI']
                    if not is_added:
                        print('Logic Wrong in L1112')
                        sys.exit()
                interval = self.save_interval(path, interval_original_center_freq, b_before, b_available, b_max,
                                              left_gb, right_gb, max_delta_low, max_delta_high, relevant_lps_lower,
                                              relevant_lps_higher, self.lightpaths_temp, self.fs_usage_temp)
                all_possible_intervals += [interval]

        # all the preparations are done, now try to serve demand with one of these intervals

        # First sort all the intervals based on first its max_available_size, then its max_delta
        all_possible_intervals = sorted(all_possible_intervals, key=lambda x: (-x[3], max(x[7], x[8])))

        # get, and rank all possible configs for the demand
        # config: [data_rate, mod_format, symbol_rate, min_FS_needed]
        # next line has potential bug if data_rate list changes
        config_ranks = defaultdict(list)
        max_provision = min(max(self.data_rates[0] // 50, math.ceil(demand / 50.0)), self.data_rates[-1] // 50)
        for i in range(max_provision, self.data_rates[0] // 50 - 1, -1):
            config_ranks[i * 50] += self.all_configs_dict[i * 50]

        for rank in config_ranks:
            for config in config_ranks[rank]:
                for interval in all_possible_intervals:
                    if interval['BandwidthAvailable'] < config[3]:
                        break
                    if not self.linear_snr_check(interval['Path'], config):
                        continue
                    self.lightpaths_temp = self.lightpath_list_deepcopy(interval['Lightpaths'])
                    self.fs_usage_temp = self.fs_usage_deepcopy(interval['FS_Usage'])
                    # find all spectrally-possible slices for the new channel
                    slices_found = self.find_possible_slice(interval['Path'], config[3], temp=True)
                    # Then sort these slices according to estimated minimal reconfiguration.
                    # The center freq will affect how many FSs are needed for the new channel
                    iocf = interval['IntervalOriginalCenterFreq']
                    slices_found = sorted(slices_found, key=lambda x: math.fabs(x[2] - iocf))
                    p_ch = self.assign_channel_power(config[2])  # launch power of the new channel

                    is_added = False
                    new_lp = None
                    for slice_found in slices_found:
                        new_lp = self.gen_lightpath(interval['Path'], config[0], config[1], config[2],
                                                    slice_found[0], slice_found[1], slice_found[2], p_ch)
                        is_added = self.add_lp(new_lp, temp=True)
                        if is_added:
                            break
                    if not is_added:
                        continue

                    # if the new channel can be established, find the reconfiguration scheme with minimum cost
                    # initialize the temporary states s1 and s2
                    lightpaths_state_1 = self.lightpaths_temp
                    fs_usage_state_1 = self.fs_usage_temp
                    lightpaths_state_2 = self.lightpath_list_deepcopy(interval['Lightpaths'])
                    fs_usage_state_2 = self.fs_usage_deepcopy(interval['FS_Usage'])

                    # find the final center frequency for channels that are reconfigured to lower spectrum
                    relevant_lps_lower = sorted(self.lightpath_list_deepcopy(interval['RelevantLightpathsLower']),
                                                key=lambda x: -(x['StartFS'] + x['SliceSize'] - 1))
                    max_shifted_distance_lower = 0
                    for i, lp in enumerate(relevant_lps_lower):
                        # get the maximum distance that lp can shift back if only considering spectrum usage
                        max_back_shift = self.hops_operation(lp, 'Right', lp['Original_CenterFreq'] - lp['CenterFreq'],
                                                             freq_unit='THz')
                        lp_backup_temp = self.lightpath_deepcopy(lp)
                        lp['Final_CenterFreq'] = lp['CenterFreq']
                        if max_back_shift == 0:
                            continue
                        for j in range(max_back_shift, 0, -1):
                            # get the current testing center frequency for the chosen lp
                            testing_center_freq = lp['CenterFreq'] + j * self.channel_spacing * 1e-3

                            # for temporary state s_1, which has the new channel already been established
                            self.lightpaths_temp = self.lightpath_list_deepcopy(lightpaths_state_1)
                            current_state_1 = self.lightpaths_temp
                            # retune the lp to the current testing center frequency
                            self.drop_lp(lp_backup_temp, temp=True)
                            lp['CenterFreq'] = testing_center_freq
                            check_pass, _ = self.add_lp(lp, temp=True)
                            if not check_pass:
                                continue

                            # for temporary state s_2, in which the new channel has NOT been established
                            self.lightpaths_temp = self.lightpath_list_deepcopy(lightpaths_state_2)
                            current_state_2 = self.lightpaths_temp
                            # retunes the lp to the current testing center frequency
                            self.drop_lp(lp_backup_temp, temp=True)
                            check_pass, _ = self.add_lp(lp, temp=True)
                            if not check_pass:
                                print('logic error at line 1174')
                                sys.exit()
                            # reset the channels that are spectrally above it to their original positions
                            reset_channel_list = list()
                            for k in range(i):
                                lp_k = relevant_lps_lower[k]
                                self.drop_lp(relevant_lps_lower[k], temp=True)
                                lp_k['CenterFreq'] = lp_k['Original_CenterFreq']
                                is_added = self.add_lp(lp_k, temp=True)
                                reset_channel_list += [lp_k]
                                if not is_added:
                                    print('LOGIC_ERROR at LINE 1183')
                                    sys.exit()

                            low_snr_flag = False
                            # test if the reconfiguration scheme is not interrupted
                            for kk, lp_k in enumerate(reset_channel_list):
                                k = len(reset_channel_list) - kk - 1
                                center_freq_shifted = lp_k['CenterFreq'] - relevant_lps_lower[k]['Final_CenterFreq']
                                spacing_to_shift = int(math.fabs(center_freq_shifted // self.channel_spacing))
                                final_center_freq = self.pp_shift_check(lp_k, spacing_to_shift, direction='Left')
                                if final_center_freq != relevant_lps_lower[k]['Final_CenterFreq']:
                                    low_snr_flag = True
                                    break
                            if low_snr_flag:
                                continue
                            lp['Final_CenterFreq'] = testing_center_freq
                            shifted_distance = lp['Original_CenterFreq'] - lp['Final_CenterFreq']
                            max_shifted_distance_lower = max(max_shifted_distance_lower, shifted_distance)
                            lightpaths_state_1 = current_state_1
                            lightpaths_state_2 = current_state_2

                    relevant_lps_higher = sorted(self.lightpath_list_deepcopy(interval['RelevantLightpathsHigher']),
                                                 key=lambda x: x['StartFS'])
                    max_shifted_distance_higher = 0
                    for i, lp in enumerate(relevant_lps_higher):
                        # get the maximum distance that lp can shift back if only considering spectrum usage
                        max_back_shift = self.hops_operation(lp, 'Left', lp['CenterFreq'] - lp['Original_CenterFreq'],
                                                             freq_unit='THz')
                        lp_backup_temp = self.lightpath_deepcopy(lp)
                        lp['Final_CenterFreq'] = lp['CenterFreq']
                        if max_back_shift == 0:
                            continue
                        for j in range(max_back_shift, 0, -1):
                            # get the current testing center frequency for the chosen lp
                            testing_center_freq = lp['CenterFreq'] - j * self.channel_spacing * 1e-3

                            # for temporary state s_1, which has the new channel already been established
                            self.lightpaths_temp = self.lightpath_list_deepcopy(lightpaths_state_1)
                            self.fs_usage_temp = self.fs_usage_deepcopy(fs_usage_state_1)
                            current_lightpaths_state_1 = self.lightpaths_temp
                            current_fs_usage_state_1 = self.fs_usage_temp
                            # retune the lp to the current testing center frequency
                            self.drop_lp(lp_backup_temp, temp=True)
                            lp['CenterFreq'] = testing_center_freq
                            check_pass, _ = self.add_lp(lp, temp=True)
                            if not check_pass:
                                continue

                            # for temporary state s_2, in which the new channel has NOT been established
                            self.lightpaths_temp = self.lightpath_list_deepcopy(lightpaths_state_2)
                            self.fs_usage_temp = self.fs_usage_deepcopy(fs_usage_state_2)
                            current_lightpaths_state_2 = self.lightpaths_temp
                            current_fs_usage_state_2 = self.fs_usage_temp
                            # retunes the lp to the current testing center frequency
                            self.drop_lp(lp_backup_temp, temp=True)
                            check_pass, _ = self.add_lp(lp, temp=True)
                            if not check_pass:
                                print('logic error at line 1251')
                                sys.exit()
                            # reset the channels that are spectrally above it to their original positions
                            reset_channel_list = list()
                            for k in range(i):
                                lp_k = relevant_lps_lower[k]
                                self.drop_lp(relevant_lps_lower[k], temp=True)
                                lp_k['CenterFreq'] = lp_k['Original_CenterFreq']
                                is_added = self.add_lp(lp_k, temp=True)
                                reset_channel_list += [lp_k]
                                if not is_added:
                                    print('LOGIC_ERROR at LINE 1262')
                                    sys.exit()

                            low_snr_flag = False
                            # test if the reconfiguration scheme is not interrupted
                            for kk, lp_k in enumerate(reset_channel_list):
                                k = len(reset_channel_list) - kk - 1
                                center_freq_shifted = relevant_lps_lower[k]['Final_CenterFreq'] - lp_k['CenterFreq']
                                spacing_to_shift = int(math.fabs(center_freq_shifted // self.channel_spacing))
                                final_center_freq = self.pp_shift_check(lp_k, spacing_to_shift, direction='Right')
                                if final_center_freq != relevant_lps_lower[k]['Final_CenterFreq']:
                                    low_snr_flag = True
                                    break
                            if low_snr_flag:
                                continue
                            lp['Final_CenterFreq'] = testing_center_freq
                            shifted_distance = lp['Final_CenterFreq'] - lp['Original_CenterFreq']
                            max_shifted_distance_higher = max(max_shifted_distance_higher, shifted_distance)
                            lightpaths_state_1 = current_lightpaths_state_1
                            fs_usage_state_1 = current_fs_usage_state_1
                            lightpaths_state_2 = current_lightpaths_state_2
                            fs_usage_state_2 = current_fs_usage_state_2

                    # 这个时候一个interval的终于算完了，存这个方案
                    minimum_cost_for_reconfig = max(max_shifted_distance_lower, max_shifted_distance_higher)
                    number_of_fs_used_along_path = len(interval['Path']) * config[3]
                    minimal_reconfig_scheme_for_interval = [lightpaths_state_1, fs_usage_state_1, new_lp['SliceSize'],
                                                            minimum_cost_for_reconfig, len(relevant_lps_lower),
                                                            len(relevant_lps_higher), number_of_fs_used_along_path]
                    available_schemes += [minimal_reconfig_scheme_for_interval]
            if available_schemes:
                reconfig_available = True
                break
        # then, pick the best scheme according to cost, numbers of lps to be reconfigured, spectral efficiency
        if available_schemes:
            available_schemes = sorted(available_schemes, key=lambda x: (x[3], x[5], -(x[4] + x[5])))
            best_scheme = available_schemes[0]
        else:
            best_scheme = None
        # if needed, make state vector for deep reinforcement learning
        ''' CODES FROM MY OLDER MODELS
        # df_state_paths = list()
        # df_intervals_info = list()
        # df_normalized_intervals_info = list()
        j_intervals = list()
        norm_j_intervals = list()
        # -------------------------- DEEP-defrag State: --------------------------
        # add infos into state after normalization
        # df state info on path k
        df_state_k = list()
        # COMMON FEATURES for EACH path in KSP:
        # feature 1. requiring FSs of Rt on the k-th path
        df_state_k.append(2 * (fs_req - 4.5) / 3.5)  # FS occupying from 1 to 8
        # feature 2,3. total available FSs and blocks on the k-th path
        total_fs_available, total_blocks = self.feat34_a_path(chosen_path)
        df_state_k.append(2 * (total_fs_available - 0.5 * self.FS_on_link) / self.FS_on_link)
        df_state_k.append(2 * (total_blocks - 0.25 * self.FS_on_link) / (0.5 * self.FS_on_link))
        # feature 4. inducing path_usage_stack: easy but effective way showing fragmentation on lp
        path_usage_stack = self.path_usage_stack(chosen_path)
        stack_count = sum(path_usage_stack)
        df_state_k.append(
            2 * (0.5 * self.FS_on_link * len(chosen_path) - stack_count) / (self.FS_on_link * len(chosen_path)))
        # INTERVALS' FEATURES - the intervals' positions in state are shuffled to strengthen the model
        if avail_intervals:  # if there's any available intervals
            is_available = True
            # print("available intervals are:", avail_intervals)
            # pick J intervals with minimum cost, and sort them in ascending order of DF cost
            avail_intervals = sorted(avail_intervals, key=(lambda an_interval: an_interval[4]))
            # get and then immediately shuffle the options to prevent that the first is always the best
            j_intervals = avail_intervals[:self.J]
            random.shuffle(j_intervals)
            # normalizations
            # return state of defrag process AFTER NORMALIZATIONS
            # state的格式目前是 [频隙位置，最大可用带宽，最佳左移，最佳右移，最佳总cost，[左移光路]，[右移光路]]
            # 最后两项是不会在最终nn的state中出现的，是拿来在后面实际调整中拿来处理的，第一项姑且先不管了
            # 至于其他项目，有点怀疑-1对于一些case是有负面影响的，例如一个较小的可用频隙可能在现在这样的state模式下会被影响其被选率
            # 所以现在需要考虑的一个问题就是如何重新量化，自己设定一个最合意的，拿来作为最佳进行量化为1如何？
            # 20.10.29: 这样是没啥意义的，因为你这就把全局的信息丢失了
            # array_max = list(np.amax(j_intervals, axis=0)[:5])
            # array_min = list(np.amin(j_intervals, axis=0)[:5])
            # print('\n', array_max, '\n', array_min)
            for interval_position in j_intervals:
                # print(interval[:5])
                temp_interval = list()
                # ------- fixed normalization -------
                # 0.频隙位置 当前 / 总频隙数
                temp0 = interval_position[0] / self.FS_on_link
                temp_interval.append(temp0)
                # 1.最大可用带宽 越小越好
                temp1 = (self.FS_on_link - interval_position[1]) / self.FS_on_link
                temp_interval.append(temp1)
                # 2.最佳左移 越小越好
                temp2 = (self.FS_on_link - interval_position[2]) / self.FS_on_link
                temp_interval.append(temp2)
                # 3.最佳右移 越小越好
                temp3 = (self.FS_on_link - interval_position[3]) / self.FS_on_link
                temp_interval.append(temp3)
                # 4.最佳总cost 越小越好
                temp4 = (self.FS_on_link - interval_position[4]) / self.FS_on_link
                temp_interval.append(temp4)
                # 5.最佳整理后中心频率
                temp5 = interval_position[5] / self.FS_on_link
                temp_interval.append(temp5)
                # 6.最大容量下中心频率
                temp6 = interval_position[6] / self.FS_on_link
                temp_interval.append(temp6)
                # print(temp_interval)
                norm_j_intervals.append(temp_interval)

            # fill j_intervals with [-1, -1 , ... , -1] until it has J elements
            # (Important) here we need to consider if -1 has any negative impact on the result （now is eliminated）
            # todo match the length of each interval state to the newest reward scheme
            while len(j_intervals) < self.J:
                insert_pos = np.random.randint(0, len(j_intervals) + 1)
                j_intervals = j_intervals[:insert_pos] + [[-1, -1, -1, -1, -1, -1, -1, [], []]] + j_intervals[
                                                                                                  insert_pos:]
                norm_j_intervals = norm_j_intervals[:insert_pos] + [
                    [-1, -1, -1, -1, -1, -1, -1]] + norm_j_intervals[
                                                    insert_pos:]
            for i in range(len(norm_j_intervals)):
                df_state_k += norm_j_intervals[i]
            # print(self.J, "intervals with minimum cost are:", j_intervals)
            # print("after normalization:", norm_j_intervals)
            # print(df_state_i)

            # Here we have the info for the intervals (original/normalized info), and flattened one for state vector
            df_intervals_info += j_intervals
            df_normalized_intervals_info += norm_j_intervals
            df_state_paths += df_state_k

        df_state = list()
        # Global_Features
        # feature 1,2. source and destination node - one hot
        src_onehot = self.node_onehot[request[0]]
        dst_onehot = self.node_onehot[request[1]]
        df_state += src_onehot  # s
        df_state += dst_onehot  # d

        # feature 3. time
        # df_state.append((self.elapsed_time) / self.time_slots_per_day)
        # feature 4. current operation's position in the training batch
        df_state.append((self.batch_size - self.pos_in_batch) / self.batch_size)

        # todo consolidate network state on each path
        df_state += df_state_paths
        df_state = np.array(df_state)  # reconstruct the output DF state
        # print("time3", end1 - start1)

        return is_available, df_intervals_info, df_normalized_intervals_info, df_state
        '''
        return reconfig_available, best_scheme

    def delta_defrag(self, lp, direction):
        # only works for temp network state, only works for 12.5/6.25 GHz
        shift_distance = self.hops_operation(lp, direction, self.FS_on_link * self.FS_bandwidth, freq_unit='GHz')
        return shift_distance

    def hops_operation(self, lp, direction, shifted_spacing, freq_unit='THz'):
        # NOTE: only works for temp network state, only works for 12.5/6.25 GHz
        if freq_unit == 'GHz':
            shifted_spacing *= 1e3
        # first get the path general FS usage
        fs_usage = self.usage_on_path(lp['Path'], temp=True)
        # print("hops ", op, direction, shift_distance)
        # print("general:\t", fs_usage)
        if direction == "Left":
            potential_fs_space = self.potential_stretch_space(fs_usage[:lp['StartFS']], direction)
        else:
            potential_fs_space = self.potential_stretch_space(fs_usage[lp['StartFS'] + lp['SliceSize']:], direction)
        if lp['SliceSize'] != math.ceil(lp['SymbolRate'] / self.FS_bandwidth):
            print('check here hops')
            potential_space = (2 * potential_fs_space + 1)
        else:
            potential_space = 2 * potential_fs_space
        return min(shifted_spacing, potential_space)

    @staticmethod
    def potential_stretch_space(link_usage, direction):
        # print(link_usage)
        temp_list = link_usage
        if direction == "Left":
            temp_list.reverse()
        if 1 in temp_list:
            next_occupied = temp_list.index(1)
            temp_list = temp_list[0:next_occupied]
        # print(len(temp_list), " slots can be moved")
        return len(temp_list)

    def pp_shift_check(self, lp, max_shifting_spacing, direction):
        if direction != 'Left' or 'Right':
            sys.exit()
        _direction = -1 if direction == 'Left' else -1

        max_spacing_shifted = 0
        low_snr_flag = False
        psd_ase = self.ase_noise_psd(lp['Path'], lp['CenterFreq'])
        minimum_snr = self.min_snr_dict[lp['ModFormat']]

        p_nli_list = list()
        final_lightpaths = self.lightpath_list_deepcopy(self.lightpaths_temp)
        for i in range(max_shifting_spacing + 1):
            # ASE noise power
            filter_extended = lp['SymbolRate'] + i * self.channel_spacing
            p_ase = psd_ase * filter_extended

            # NLI noise power of channel at frequencies that has been calculated before
            for j in range(i):
                p_nli = p_nli_list[j]
                snr_db = 10 * math.log10(lp['LaunchPower'] / (p_ase + p_nli))
                if snr_db < minimum_snr:
                    low_snr_flag = True
                    break
            if low_snr_flag:
                self.lightpaths_temp = final_lightpaths
                break

            # NLI noise power of channel at current most distant center frequency.
            current_furthest_center_freq_shifted = _direction * i * self.channel_spacing * 1e-3
            current_furthest_center_freq = lp['CenterFreq'] + current_furthest_center_freq_shifted
            # 1. run cross_check, remove the NLI impact of this lp at its original center frequency on other lps.
            self.cross_snr_check_and_update(lp['Path'], lp, add_or_drop='Drop', temp=True, update=True)
            # self.cross_snr_check_and_update(lp['Path'], lp, direction=direction, add_or_drop='Drop',
            #                                 temp=True, update=True)
            # 2. get the p_nli of the lp when at this frequency:
            lp['CenterFreq'] = current_furthest_center_freq
            self_check_pass, p_ase, p_nli = self.self_check(lp['Path'], lp['ModFormat'], lp['SymbolRate'],
                                                            lp['CenterFreq'], lp['LaunchPower'],
                                                            filter_extended=filter_extended, temp=True)
            lp['ASE'] = p_ase
            lp['NLI'] = p_nli
            snr_db = 10 * math.log10(lp['LaunchPower'] / (p_ase + p_nli))
            if snr_db < minimum_snr:
                self.lightpaths_temp = final_lightpaths
                break
            # 3. if passed, run cross_check to ensure the shifted lp won't affect the functioning of the other lps.
            if self_check_pass:
                xcheck_pass = self.cross_snr_check_and_update(lp['Path'], lp, add_or_drop='Add', temp=True, update=True)
                # xcheck_pass = self.cross_snr_check_and_update(lp['Path'], lp, direction=direction, add_or_drop='Add',
                #                                               temp=True, update=True)
                if xcheck_pass:
                    # 4. finally update the current position as the best, then go for the next
                    max_spacing_shifted = i
                    final_lightpaths = self.lightpath_list_deepcopy(self.lightpaths_temp)
                    p_nli_list += [p_nli]
                else:
                    self.lightpaths_temp = final_lightpaths
                    break
        final_center_freq = lp['Original_CenterFreq'] + _direction * max_spacing_shifted * self.channel_spacing * 1e-3
        return final_center_freq, lp

    def filter_lps_on_link(self, link, target_fs_pos, direction, temp=False):
        link_lps_dict_using = self.link_lps_dict_temp if temp else self.link_lps_dict
        if direction == 'Left':
            return [lp for lp in link_lps_dict_using[link[0] * 100 + link[1]] if lp['StartFS'] < target_fs_pos]
        elif direction == 'Right':
            return [lp for lp in link_lps_dict_using[link[0] * 100 + link[1]] if lp['StartFS'] > target_fs_pos]
        else:
            return [lp for lp in link_lps_dict_using[link[0] * 100 + link[1]] if lp['StartFS'] != target_fs_pos]

    def get_origin_lightpath(self, lp):
        lp_original = self.lightpath_deepcopy(lp)
        lp_original['StartFS'] = lp['Original_StartFS']
        lp_original['SliceSize'] = lp['Original_SliceSize']
        lp_original['CenterFreq'] = lp['Original_CenterFreq']
        lp_original['ASE'] = None
        lp_original['NLI'] = None
        return lp_original

    def interval_original_center_freq(self, f_before, c_before, b_before):
        if b_before > 0:
            return self.get_center_freq(f_before + 1, b_before)
        elif b_before == 0:
            return self.get_center_freq(f_before + 1, b_before)
        else:
            return self.get_center_freq(c_before, f_before - c_before + 1)

    @staticmethod
    def save_interval(path, interval_original_center_freq, b_before, b_available, b_max, left_gb, right_gb,
                      max_delta_low, max_delta_high, relevant_lps_lower, relevant_lps_higher, lightpaths, fs_usage):
        interval_info = dict()
        interval_info['Path'] = path
        interval_info['IntervalOriginalCenterFreq'] = interval_original_center_freq
        interval_info['BandwidthBefore'] = b_before
        interval_info['BandwidthAvailable'] = b_available
        interval_info['BandwidthMax'] = b_max
        interval_info['LeftGuardband'] = left_gb
        interval_info['RightGuardband'] = right_gb
        interval_info['MaxDeltaLow'] = max_delta_low
        interval_info['MaxDeltaHigh'] = max_delta_high
        interval_info['RelevantLightpathsLower'] = relevant_lps_lower
        interval_info['RelevantLightpathsHigher'] = relevant_lps_higher
        interval_info['AllLightpaths'] = lightpaths
        interval_info['FS_Usage'] = fs_usage
        return interval_info

    # ---------------------------------- Utils - EON -----------------------------------------------------------------
    @staticmethod
    def gen_lightpath(path, data_rate, mod_format, symbol_rate, start_fs, slice_size,
                      center_freq, p_ch, p_ase=None, p_nli=None):
        lightpath = defaultdict()
        lightpath['Path'] = path
        lightpath['DataRate'] = data_rate  # in GBaud
        lightpath['ModFormat'] = mod_format
        lightpath['SymbolRate'] = symbol_rate  # in GHz
        lightpath['StartFS'] = start_fs
        lightpath['SliceSize'] = slice_size
        lightpath['CenterFreq'] = center_freq  # in THz
        lightpath['LaunchPower'] = p_ch  # in Watts
        if p_ase:
            lightpath['ASE'] = p_ase
        if p_nli:
            lightpath['NLI'] = p_nli
        return lightpath

    def assign_channel_power(self, symbol_rate):
        # note here the PSD of each channel is fixed, because i found that the eta_nli in the ECOC paper
        # is power-independent only if all channel have the same launch power / power spectrum density.
        return self.global_signal_psd * symbol_rate * 1e-3

    def freq_2_fs(self, center_freq):
        return (center_freq - self.f_criterion) / 1e-3 / self.FS_bandwidth + 0.5 * self.FS_on_link  # in FS

    def add_lp(self, new_lp, temp=False):
        lp_list_using = self.lightpaths_temp if temp else self.lightpaths
        self_check_pass, p_ase, p_nli = self.self_check(new_lp['Path'], new_lp['ModFormat'], new_lp['SymbolRate'],
                                                        new_lp['CenterFreq'], new_lp['LaunchPower'], temp=temp)
        new_lp['ASE'] = p_ase
        new_lp['NLI'] = p_nli
        if self_check_pass:
            # run cross check, make sure the target lightpath won't affect the functioning of the other lps.
            xcheck_pass = self.cross_snr_check_and_update(new_lp['Path'], new_lp, add_or_drop='Add', temp=temp)
            if xcheck_pass:
                lp_list_using += [new_lp]
                add_pass = self.take_fs(new_lp['Path'], new_lp['StartFS'], new_lp['SliceSize'], temp=temp)
                if add_pass:
                    return True, new_lp
                else:
                    print('Error add_lp')
        return False, new_lp

    def drop_lp(self, lp2drop, temp=False):
        lp_list_using = self.lightpaths_temp if temp else self.lightpaths
        if lp2drop not in lp_list_using:
            print('drop_lp error: lp not in list!')
            sys.exit()
        xcheck_pass = self.cross_snr_check_and_update(lp2drop['Path'], lp2drop, add_or_drop='Drop', temp=temp)
        if xcheck_pass:
            lp_list_using.remove(lp2drop)
            drop_pass = self.release_fs(lp2drop['Path'], lp2drop['StartFS'], lp2drop['SliceSize'], temp=temp)
            if drop_pass:
                return True
            else:
                print('Error drop_lp')
        return False

    def shared_span_cut(self, path_cut, path_nch):
        links_cut = self.path2link(path_cut)
        links_nch = self.path2link(path_nch)
        span_shared = list()
        span_count = 0
        for i, link_cut_i in enumerate(links_cut):
            spans_on_i = self.link_spans_dict[link_cut_i[0] * 100 + link_cut_i[1]]
            for link_nch_j in links_nch:
                if link_cut_i == link_nch_j or link_cut_i == link_nch_j[::-1]:
                    span_shared += [[ii, span_count + jj] for ii in spans_on_i for jj in range(len(spans_on_i))]
            span_count += len(self.link_spans_dict[link_cut_i[0] * 100 + link_cut_i[1]])
        return span_shared  # [[span, span_index for cut] ,...]

    def usage_on_path(self, path, temp=False):
        links_on_path = self.path2link(path)
        usage_on_path = np.array([0] * self.FS_on_link)
        if temp:
            for link in links_on_path:
                usage_on_path += np.array(self.fs_usage_temp[link[0] * 100 + link[1]])
        else:
            for link in links_on_path:
                usage_on_path += np.array(self.fs_usage[link[0] + link[1] * 100])
        usage_on_path = np.minimum(usage_on_path, 1)
        return list(usage_on_path)

    def update_link_lps_dict(self, temp=False):
        # make a dictionary for saving FS usage on every path
        # from serviceQueue get all the links' usage. save it in link_lps_dict
        # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
        new_link_lps_dict = defaultdict(lambda: list())
        lp_list = self.lightpaths_temp if temp else self.lightpaths
        for lp in lp_list:
            path = lp['Path']
            links_using = self.path2link(path)
            for link in links_using:
                lp_get = self.lightpath_deepcopy(lp)
                new_link_lps_dict[link[0] * 100 + link[1]] += [lp_get]
                new_link_lps_dict[link[1] * 100 + link[0]] += [lp_get]
        if temp:
            self.link_lps_dict_temp = new_link_lps_dict
        else:
            self.link_lps_dict = new_link_lps_dict

    def update_path_lps_dict(self):
        # make a dictionary for saving FS usage on every path
        # from serviceQueue get all the links' usage. save it in link_lps_dict
        self.path_lps_dict = defaultdict(lambda: list())
        for lp in self.lightpaths:
            lp_get = self.lightpath_deepcopy(lp)
            self.path_lps_dict[str(lp['Path'])] += [lp_get]
            self.path_lps_dict[str(lp['Path'][::-1])] += [lp_get]

    def update_res_fs(self, temp=False):
        # use all_lps to generate new res_fs
        node_num = len(self.distance_matrix)
        x = np.array([[x] * node_num for x in range(node_num)]).flatten()
        y = list(range(node_num)) * node_num
        if temp:
            lp_list = self.lightpaths_temp
            self.fs_usage_temp = dict()
        else:
            lp_list = self.lightpaths
            self.fs_usage = dict()
        get_dict = map(lambda a, b: self.new_dict(a, b, temp), x, y)
        list(get_dict)
        for lp in lp_list:
            ret = self.take_fs(lp['Path'], lp['StartFS'], lp['BlockSize'], temp=temp)
            if not ret:
                print("ERROR 703")

    def new_dict(self, x0, y0, temp):
        dict_to_op = self.fs_usage_temp if temp else self.fs_usage
        if self.distance_matrix[x0][y0] > 0:
            dict_to_op[x0 * 100 + y0] = self.FS_on_link * [0]

    def find_slice(self, path, target_size, slice2find=float('Inf'), temp=False):
        slice_index = list()
        path_usage = self.usage_on_path(path, temp)
        pointer = 0
        while 0 in path_usage:
            start_fs = path_usage.index(0)
            pointer += start_fs
            path_usage = path_usage[start_fs:]
            if 1 in path_usage:
                slice_size = path_usage.index(1)
            else:
                slice_size = self.FS_on_link - pointer
            if slice_size >= target_size:
                slice_index += [pointer + i for i in range(slice_size - target_size + 1)]
                if len(slice_index) >= slice2find:
                    break
            if pointer + slice_size == self.FS_on_link:
                break
            else:
                path_usage = path_usage[slice_size:]
                pointer += slice_size
        if len(slice_index) > slice2find:
            return slice_index[:slice2find]
        else:
            return slice_index

    # for push-pull
    def find_possible_slice(self, path, target_size, temp=False):
        slice_found = list()
        path_usage = self.usage_on_path(path, temp)
        p = 0  # pointer
        while 0 in path_usage:
            start_fs = path_usage.index(0)
            p += start_fs
            path_usage = path_usage[start_fs:]
            if 1 in path_usage:
                slice_size = path_usage.index(1) - 1
            else:
                slice_size = self.FS_on_link - p
            if slice_size >= target_size:
                slice_found += [[p + i, target_size, self.center_freq_dict[str([p + i, target_size])]]
                                for i in range(slice_size - target_size + 1)]
                slice_found += [[p + i, target_size + 1, self.center_freq_dict[str([p + i, target_size + 1])]]
                                for i in range(slice_size - (target_size + 1) + 1)]
            if p + slice_size == self.FS_on_link:
                break
            else:
                path_usage = path_usage[slice_size:]
                p += slice_size
        return slice_found

    def release_fs(self, path, start_fs, slice_size, temp=False):
        # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
        ret = True
        link_on_path = self.path2link(path)
        fs_usage_dict = self.fs_usage_temp if temp else self.fs_usage
        for fs in range(start_fs, start_fs + slice_size):
            for link in link_on_path:
                if fs_usage_dict[link[0] * 100 + link[1]][fs] == 0:
                    print("error in release_fs")
                    ret = False
                else:
                    fs_usage_dict[link[0] * 100 + link[1]][fs] = 0
                    fs_usage_dict[link[1] * 100 + link[0]][fs] = 0
        return ret

    def take_fs(self, path, start_fs, slice_size, temp=False):
        # lp: [path, data_rate, mod_format, symbol_rate, start_fs, slice_size, center_freq, p_ch]
        ret = True
        link_on_path = self.path2link(path)
        fs_usage_dict = self.fs_usage_temp if temp else self.fs_usage
        for fs in range(start_fs, start_fs + slice_size):
            for link in link_on_path:
                if fs_usage_dict[link[0] * 100 + link[1]][fs] == 1:
                    print("error in take_fs")
                    ret = False
                else:
                    fs_usage_dict[link[0] * 100 + link[1]][fs] = 1
                    fs_usage_dict[link[1] * 100 + link[0]][fs] = 1
        return ret

    def get_center_freq(self, start_fs, slice_size):
        if str([start_fs, slice_size]) in self.center_freq_dict:
            return self.center_freq_dict[str([start_fs, slice_size])]
        else:
            for start_fs_i in range(self.FS_on_link):
                if start_fs_i + slice_size > self.FS_on_link:
                    break
                center_freq_i = self.f_criterion + (
                        start_fs_i + 0.5 * slice_size - 0.5 * self.FS_on_link) * self.FS_bandwidth * 1e-3  # in THz
                self.center_freq_dict[str([start_fs_i, slice_size])] = center_freq_i
            return self.center_freq_dict[str([start_fs, slice_size])]

    def path2link(self, path):
        path = tuple(path)
        if path in self.path_dict:
            # print("path in path_dict:", path)
            links_on_path = self.path_dict[path]
        else:
            links_on_path = list()
            for nodeA in range(len(path[:-1])):
                links_on_path.append([path[nodeA], path[nodeA + 1]])
            self.path_dict[path] = links_on_path
        return links_on_path

    def path2span(self, path):
        path = tuple(path)
        if path in self.path2span_dict:
            return self.path2span_dict[path]
        else:
            spans_on_path = []
            links_on_path = self.path2link(path)
            for link in links_on_path:
                spans_on_link = self.link_spans_dict[link[0] * 100 + link[1]]
                for span in spans_on_link:
                    spans_on_path += [span]
            self.path2span_dict[path] = spans_on_path
            return spans_on_path

    def find_span_index_on_path(self, path, span):
        spans_on_path = self.path2span(path)
        return spans_on_path.index(span)

    # ------------------------------------ Utils - Get fiber parameters ------------------------------------

    # in db
    def attenuation(self, span_type):
        return self.fiber_parameters[span_type]['attnDB']

    # in ps2/km
    def dispersion(self, span_type):
        return self.fiber_parameters[span_type]['Dispersion']

    # in ps3/km
    def dispersion_slope(self, span_type):
        return self.fiber_parameters[span_type]['DispersionSlope']

    # in 1/(W·km)
    def fnlc(self, span_type):
        return self.fiber_parameters[span_type]['FiberNonLinearityCoefficient']

    # ------------------------------------ Utils - deepcopy -----------------------------------------------

    # 1 layer "deep" copy
    @staticmethod
    def lightpath_list_deepcopy(lp_list):
        list_copy = list()
        for lp in lp_list:
            copy = dict()
            for key in lp:
                if key == 'Path':
                    _copy = lp[key][:]
                else:
                    _copy = lp[key]
                copy[key] = _copy
            list_copy = [copy]
        return list_copy

    @staticmethod
    def lightpath_deepcopy(lp):
        copy = dict()
        for key in lp:
            if key == 'Path':
                _copy = lp[key][:]
            else:
                _copy = lp[key]
            copy[key] = _copy
        return copy

    @staticmethod
    def lightpath_update(paste_to, copy_from):
        for key in copy_from:
            if key == 'Path':
                paste_to[key][:] = copy_from[key][:]
            else:
                paste_to[key] = copy_from[key]

    # 1 layer "deep" copy
    @staticmethod
    def fs_usage_deepcopy(res_fs):
        copy = dict()
        for res in res_fs:
            copy[res] = res_fs[res][:]
        return copy


if __name__ == "__main__":
    eon_env = NetworkEnv()
    eon_env.ecoc_sim()
    print("OK")
