//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "mmvq.hpp"
#include "vecdotq.hpp"

typedef float (*vec_dot_q_sycl_t)(
    const void* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    const int& iqs);

template <
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint32_t* iq3xxs_grid_ptr,
    const uint64_t* ksigns64_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint64_t* iq2xxs_grid_ptr,
    const uint8_t* ksigns_iq2xs_ptr,
    const uint8_t* kmask_iq2xs_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp += vec_dot_iq2_xxs_q8_1(
        &x[ibx],
        &y[iby],
        iqs,
        iq2xxs_grid_ptr,
        ksigns_iq2xs_ptr,
        kmask_iq2xs_ptr);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint64_t* iq2xs_grid_ptr,
    const uint64_t* ksigns64_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp += vec_dot_iq2_xs_q8_1(
        &x[ibx], &y[iby], iqs, iq2xs_grid_ptr, ksigns64_ptr);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint32_t* iq3xxs_grid_ptr,
    const uint64_t* ksigns64_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp += vec_dot_iq3_xxs_q8_1(
        &x[ibx], &y[iby], iqs, iq3xxs_grid_ptr, ksigns64_ptr);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint32_t* iq3s_grid_ptr,
    const uint64_t* ksigns64_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp +=
        vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid_ptr, ksigns64_ptr);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    float* __restrict__ dst,
    const int ncols,
    const int nrows,
    const sycl::nd_item<3>& item_ct1,
    const uint32_t* iq1s_grid_ptr,
    const uint64_t* ksigns64_ptr) {
  const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
      item_ct1.get_local_id(1);

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i; // x block index

    const int iby = i * (qk / QK8_1); // y block index that aligns with ibx

    const int iqs = vdr *
        (item_ct1.get_local_id(2) %
         (qi / vdr)); // x block quant index when casting the quants to int

    tmp +=
        vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_ptr, ksigns64_ptr);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
  }

  if (item_ct1.get_local_id(2) == 0) {
    dst[row] = tmp;
  }
}

static void mul_mat_vec_q4_0_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK4_0 == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK4_0,
                QI4_0,
                block_q4_0,
                VDR_Q4_0_Q8_1_MMVQ,
                vec_dot_q4_0_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q4_1_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK4_1 == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK4_0,
                QI4_1,
                block_q4_1,
                VDR_Q4_1_Q8_1_MMVQ,
                vec_dot_q4_1_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q5_0_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK5_0 == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK5_0,
                QI5_0,
                block_q5_0,
                VDR_Q5_0_Q8_1_MMVQ,
                vec_dot_q5_0_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q5_1_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK5_1 == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK5_1,
                QI5_1,
                block_q5_1,
                VDR_Q5_1_Q8_1_MMVQ,
                vec_dot_q5_1_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q8_0_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK8_0 == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK8_0,
                QI8_0,
                block_q8_0,
                VDR_Q8_0_Q8_1_MMVQ,
                vec_dot_q8_0_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q2_K_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK_K,
                QI2_K,
                block_q2_K,
                VDR_Q2_K_Q8_1_MMVQ,
                vec_dot_q2_K_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q3_K_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK_K,
                QI3_K,
                block_q3_K,
                VDR_Q3_K_Q8_1_MMVQ,
                vec_dot_q3_K_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q4_K_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK_K,
                QI4_K,
                block_q4_K,
                VDR_Q4_K_Q8_1_MMVQ,
                vec_dot_q4_K_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q5_K_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK_K,
                QI5_K,
                block_q5_K,
                VDR_Q5_K_Q8_1_MMVQ,
                vec_dot_q5_K_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_q6_K_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q<
                QK_K,
                QI6_K,
                block_q6_K,
                VDR_Q6_K_Q8_1_MMVQ,
                vec_dot_q6_K_q8_1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_iq2_xxs_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq2xxs_grid.init(*stream);
    ksigns_iq2xs.init(*stream);
    kmask_iq2xs.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq2xxs_grid_ptr_ct1 = iq2xxs_grid.get_ptr();
      auto ksigns_iq2xs_ptr_ct1 = ksigns_iq2xs.get_ptr();
      auto kmask_iq2xs_ptr_ct1 = kmask_iq2xs.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS, block_iq2_xxs, 1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq2xxs_grid_ptr_ct1,
                ksigns_iq2xs_ptr_ct1,
                kmask_iq2xs_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq2xs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq2xs_grid_ptr_ct1 = iq2xs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS, block_iq2_xs, 1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq2xs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3xxs_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3xxs_grid_ptr_ct1 = iq3xxs_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS, block_iq3_xxs, 1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3xxs_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq3s_grid.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq3s_grid_ptr_ct1 = iq3s_grid.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_XS, block_iq3_s, 1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq3s_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(
    const void* vx,
    const void* vy,
    float* dst,
    const int ncols,
    const int nrows,
    dpct::queue_ptr stream) {
  GGML_ASSERT(ncols % QK_K == 0);
  const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
  const sycl::range<3> block_nums(1, 1, block_num_y);
  const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
  {
    iq1s_grid_gpu.init(*stream);
    ksigns64.init(*stream);

    stream->submit([&](sycl::handler& cgh) {
      auto iq1s_grid_ptr_ct1 = iq1s_grid_gpu.get_ptr();
      auto ksigns64_ptr_ct1 = ksigns64.get_ptr();

      cgh.parallel_for(
          sycl::nd_range<3>(block_nums * block_dims, block_dims),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                vx,
                vy,
                dst,
                ncols,
                nrows,
                item_ct1,
                iq1s_grid_ptr_ct1,
                ksigns64_ptr_ct1);
          });
    });
  }
}

void ggml_sycl_op_mul_mat_vec_q(
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst,
    const char* src0_dd_i,
    const float* src1_ddf_i,
    const char* src1_ddq_i,
    float* dst_dd_i,
    const int64_t row_low,
    const int64_t row_high,
    const int64_t src1_ncols,
    const int64_t src1_padded_row_size,
    const dpct::queue_ptr& stream) {
  GGML_ASSERT(ggml_nrows(src1) == 1);

  const int64_t ne00 = src0->ne[0];
  const int64_t row_diff = row_high - row_low;

  switch (src0->type) {
    case GGML_TYPE_Q4_0:
      mul_mat_vec_q4_0_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q4_1:
      mul_mat_vec_q4_1_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q5_0:
      mul_mat_vec_q5_0_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q5_1:
      mul_mat_vec_q5_1_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q8_0:
      mul_mat_vec_q8_0_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q2_K:
      mul_mat_vec_q2_K_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q3_K:
      mul_mat_vec_q3_K_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q4_K:
      mul_mat_vec_q4_K_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q5_K:
      mul_mat_vec_q5_K_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_Q6_K:
      mul_mat_vec_q6_K_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      mul_mat_vec_iq2_xxs_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      mul_mat_vec_iq2_xs_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      mul_mat_vec_iq3_xxs_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_IQ3_S:
      mul_mat_vec_iq3_s_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    case GGML_TYPE_IQ1_S:
      mul_mat_vec_iq1_s_q8_1_sycl(
          src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stream);
      break;
    default:
      GGML_ASSERT(false);
      break;
  }

  (void)src1;
  (void)dst;
  (void)src1_ddf_i;
  (void)src1_ncols;
  (void)src1_padded_row_size;
}
