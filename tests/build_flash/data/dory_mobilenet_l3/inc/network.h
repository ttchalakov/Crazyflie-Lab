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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <stddef.h>
#include "pmsis.h"


struct network_run_token {
  struct pi_device cluster_dev;
};


void network_terminate();
void network_initialize();
void network_run_cluster(void * args);
struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void network_run_wait(struct network_run_token token);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  "ReluConvolution0_weights.hex", "ReluConvolution1_weights.hex", "ReluConvolution3_weights.hex", "ReluConvolution4_weights.hex", "ReluConvolution6_weights.hex", "ReluConvolution7_weights.hex", "FullyConnected10_weights.hex"
};
static int L3_weights_size[7];
static int layers_pointers[11];
static char * Layers_name[11] = {"ReluConvolution0", "ReluConvolution1", "Pooling2", "ReluConvolution3", "ReluConvolution4", "Pooling5", "ReluConvolution6", "ReluConvolution7", "Pooling8", "Pooling9", "FullyConnected10"};
static int L3_input_layers[11] = {1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[11] = {1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1};
static int branch_input[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[11] = {15852, 54848, 0, 29491, 235471, 0, 65035, 702320, 0, 0, 43662};
static int weights_size[11] = {1168, 640, 0, 2336, 2304, 0, 4672, 6528, 0, 0, 400};
static int activations_checksum[11][1] = {{
  1173946  },
{
  296597  },
{
  745237  },
{
  339574  },
{
  124431  },
{
  341653  },
{
  131377  },
{
  33625  },
{
  96748  },
{
  36814  },
{
  4056  }
};
static int activations_size[11] = {9216, 9216, 18432, 4608, 4608, 9216, 2304, 2304, 3456, 864, 96};
static int out_mult_vector[11] = {2061, 2027, 1, 1786, 1857, 1, 1103, 1819, 1, 1, 1};
static int out_shift_vector[11] = {14, 14, 0, 14, 14, 0, 14, 14, 0, 0, 0};
static int activations_out_checksum[11][1] = {{
  296597 },
{
  745237 },
{
  339574 },
{
  124431 },
{
  341653 },
{
  131377 },
{
  33625 },
{
  96748 },
{
  36814 },
{
  4056 },
{
  2732 }
};
static int activations_out_size[11] = {9216, 18432, 4608, 4608, 9216, 2304, 2304, 3456, 864, 96, 16};
static int layer_with_weights[11] = {1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1};
#endif

#endif  // __NETWORK_H__
