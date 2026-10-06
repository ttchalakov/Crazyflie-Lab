/*
 * network.h
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#ifndef __POLICY_NETWORK_H__
#define __POLICY_NETWORK_H__

#include "policy_weights_definition.h"
#include <stddef.h>
#include "pmsis.h"


struct policy_network_run_token {
  struct pi_device cluster_dev;
};


void policy_network_run_cluster(void * args);
struct policy_network_run_token policy_network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h);
void policy_network_run_wait(struct policy_network_run_token token);
void policy_network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h);
void policy_execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
static char * Layers_name[3] = {"policy_ReluFullyConnected0", "policy_ReluFullyConnected1", "policy_FullyConnected2"};
static char *Weights_name[3] = {Weights_policy_ReluFullyConnected0, Weights_policy_ReluFullyConnected1, Weights_policy_FullyConnected2};
static int branch_input[3] = {0, 0, 0};
static int branch_output[3] = {0, 0, 0};
static int branch_change[3] = {0, 0, 0};
static int weights_checksum[3] = {120184, 521725, 33506};
static int weights_size[3] = {960, 4352, 272};
static int activations_checksum[3][1] = {{
  1316  },
{
  3176  },
{
  3013  }
};
static int activations_size[3] = {11, 64, 64};
static int out_mult_vector[3] = {945, 832, 1};
static int out_shift_vector[3] = {16, 17, 0};
static int activations_out_checksum[3][1] = {{
  3176 },
{
  3013 },
{
  1594 }
};
static int activations_out_size[3] = {64, 64, 16};
static int layer_with_weights[3] = {1, 1, 1};
#endif

#endif  // __NETWORK_H__
