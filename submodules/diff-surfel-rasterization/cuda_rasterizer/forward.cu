/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include<iostream>
#include <cuda_runtime.h>
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <device_functions.h> 
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec2 scale,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, 1.0f);
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

#if DUAL_VISIABLE
	float multiplier = normal.z < 0 ? 1: -1;
	normal = multiplier * normal;
#endif
}

// Computing the bounding box of the 2D Gaussian and its center
// The center of the bounding box is used to create a low pass filter
__device__ bool compute_aabb(
	glm::mat3 T, 
	float2& point_image,
	float2 & extent
) {
	float3 T0 = {T[0][0], T[0][1], T[0][2]};
	float3 T1 = {T[1][0], T[1][1], T[1][2]};
	float3 T3 = {T[2][0], T[2][1], T[2][2]};

	// Compute AABB
	float3 temp_point = {1.0f, 1.0f, -1.0f};
	float distance = sumf3(T3 * T3 * temp_point);
	float3 f = (1 / distance) * temp_point;
	if (distance == 0.0) return false;

	point_image = {
		sumf3(f * T0 * T3),
		sumf3(f * T1 * T3)
	};  
	
	float2 temp = {
		sumf3(f * T0 * T0),
		sumf3(f * T1 * T1)
	};
	float2 half_extend = point_image * point_image - temp;
	extent = sqrtf2(maxf2(1e-4, half_extend));
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
    
    //if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0)printf("successPro ");
    
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
    
    
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}
    
    //if(idx==6523)printf("exist\n");
    
	// Compute center and radius
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, point_image, extent);
		if (!ok) return;
		radius = 3.0f * ceil(max(extent.x, extent.y));
	}
    
    //if(idx==6523)printf("radius%f\n",radius);
    //if(idx==6523)printf("px%f py%f\n",point_image.x,point_image.y);
	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
    //if(idx==6523)printf("rgb%f\n",rgb[idx * C + 0]);
    //if(idx==6523)printf("rmaxx%d rmaxy%d rminx%d rminy%d \n",rect_max.x,rect_max.y,rect_min.x,rect_min.y);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;
    if(idx==6523)printf("exist2\n");
	// Compute colors 
	if (colors_precomp == nullptr) {
        
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
        //if(idx==6523)printf("rgbbef%f\n",rgb[idx * C + 0]);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
        //if(idx==6523)printf("rgbafter%f\n",rgb[idx * C + 0]);
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
    
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const int* trian,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{

    if(blockIdx.x==3&&blockIdx.y==3&&threadIdx.x==3&&threadIdx.y==3)printf("transMats%f\n",transMats[9*7678]);
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
    
    
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
    //if(blockIdx.x==3&&blockIdx.y==3&&threadIdx.x==3&&threadIdx.y==3)printf("BLOCK_SIZE%d\n",BLOCK_SIZE);
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
    
    //if(blockIdx.x==20&&blockIdx.y==20)printf("todo%d\n",toDo);

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE][3];
    __shared__ int collected_id_tri[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE][3];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE][3];
	__shared__ float3 collected_Tu[BLOCK_SIZE][3];
	__shared__ float3 collected_Tv[BLOCK_SIZE][3];
	__shared__ float3 collected_Tw[BLOCK_SIZE][3];
    
    //if(blockIdx.x==3&&blockIdx.y==3&&threadIdx.x==3&&threadIdx.y==3)printf("pz%f\n",collected_Tw[0][0].x);


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
    //if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0)printf("buildsuccess626 ");
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
        if(i==0&&blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0){
        //for(int j=0;j<7000;j++)printf("opacity %f\n",normal_opacity[trian[3*j]].w);
        }
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int triangle = point_list[range.x + progress];
            //if(block.group_index().x==0&&block.group_index().y==0)printf("triangle%d ",triangle);
            
            for(int j=0;j<3;j++){
            int pid=trian[3*triangle+j];
			collected_id[block.thread_rank()][j] = pid;
            //collected_id_tri[block.thread_rank()]=gs2tri[pid];
			collected_xy[block.thread_rank()][j] = points_xy_image[pid];
            //if(blockIdx.x==20&&blockIdx.y==20)printf("pid%d nm_pid%f\n",pid,normal_opacity[pid].w);
			collected_normal_opacity[block.thread_rank()][j] = normal_opacity[pid];
			collected_Tu[block.thread_rank()][j] = {transMats[9 * pid+0], transMats[9 * pid+1], transMats[9 * pid+2]};
			collected_Tv[block.thread_rank()][j] = {transMats[9 * pid+3], transMats[9 * pid+4], transMats[9 * pid+5]};
			collected_Tw[block.thread_rank()][j] = {transMats[9 * pid+6], transMats[9 * pid+7], transMats[9 * pid+8]};
            }
            //if(block.thread_rank()==0)printf("\n");
		}
		block.sync();
        //if(blockIdx.x==3&&blockIdx.y==3&&threadIdx.x==3&&threadIdx.y==3)printf("pz%f\n",transMats[9000+6]);
        
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
            
            float alphatri=0;
            float alpha[3];
            float w[3];
            int tag=1;
            for(int vt=0;vt<3;vt++){
                // Fisrt compute two homogeneous planes, See Eq. (8)
                const float2 xy = collected_xy[j][vt];
                const float3 Tu = collected_Tu[j][vt];
                const float3 Tv = collected_Tv[j][vt];
                const float3 Tw = collected_Tw[j][vt];
                float3 k = pix.x * Tw - Tu;
                float3 l = pix.y * Tw - Tv;
                float3 p = cross(k, l);
                
                //if(j<5&&blockIdx.x==20&&blockIdx.y==20)printf("k%f l%f\n",k,l);
                
                if (p.z == 0.0){
                    tag=0;
                    continue;
                }
                float2 s = {p.x / p.z, p.y / p.z};
                float rho3d = (s.x * s.x + s.y * s.y); 
                float2 d = {xy.x - pixf.x, xy.y - pixf.y};
                float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

                // compute intersection and depth
                float rho = min(rho3d, rho2d);
                float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
                if (depth < near_n){
                    tag=0;
                    continue;
                }
                float4 nor_o = collected_normal_opacity[j][vt];
                float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
                float opa = nor_o.w;

                float power = -0.5f * rho;
                
                //if(blockIdx.x==15&&blockIdx.y==15&&rho<1)printf("rho%f\n",rho);
                //if(j<5&&blockIdx.x==20&&blockIdx.y==20)printf("opa%f power%f\n",opa,power);
                if (power > 0.0f){
                    tag=0;
                    continue;
                }

                // Eq. (2) from 3D Gaussian splatting paper.
                // Obtain alpha by multiplying with Gaussian opacity
                // and its exponential falloff from mean.
                // Avoid numerical instabilities (see paper appendix). 
                alpha[vt] = min(0.99f, opa * exp(power));
                alphatri+=alpha[vt]/3;
                w[vt]=alpha[vt]/3*T;
                
                
                
                
                // Render normal map
                for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w[vt];
                
                #if RENDER_AXUTILITY
                // Render depth distortion map
                // Efficient implementation of distortion loss, see 2DGS' paper appendix.
                float A = 1-T;
                float m = far_n / (far_n - near_n) * (1 - near_n / depth);
                distortion += (m * m * A + M2 - 2 * m * M1) * w[vt];
                D  += depth * w[vt];
                M1 += m * w[vt];
                
                float dat=m * m * w[vt];
                M2 += dat;
                
                /*
                if(isnan(M2)){
                    printf("pid%d\n",pix_id);
                    if(isnan(dat))printf("dat-1\n");
                    if(isnan(m))printf("m-1\n");
                    if(isnan( w[vt]))printf(" w[vt]-1\n");
                    //if(isnan(M2))printf("M2-1\n");
                    if(isnan(near_n / depth))printf("near_n%f depth%f mb-1\n",near_n,depth);
                }
                */
                if (T > 0.5) {
                    median_depth = depth;
                    // median_weight = w[vt];
                    median_contributor = contributor;
                }

                #endif
                //if(blockIdx.x==15&&blockIdx.y==15)printf("w%f\n",w[vt]);
            
            }
            
            //if(blockIdx.x==15&&blockIdx.y==15&&alphatri>0.01)printf("alphatri%f\n",alphatri);
            
			if (alphatri < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alphatri);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
            if(!tag)continue;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++){
                for(int vt=0;vt<3;vt++)
                {
                    C[ch] += features[collected_id[j][vt] * CHANNELS + ch] * w[vt];
                    //if(blockIdx.x==15&&blockIdx.y==15)printf("C%f\n",C[ch]);
                }
             }
            
            //if(blockIdx.x==37&&blockIdx.y==36&&threadIdx.x==15&&threadIdx.y==3)printf("features%f\n",features[collected_id[j][0] * CHANNELS]);
            //if(blockIdx.x==37&&blockIdx.y==36&&threadIdx.x==15&&threadIdx.y==3)printf("collected_id%d\n",collected_id[j][0]);
            
            
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
        {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
            //if(blockIdx.x==15&&blockIdx.y==15)printf("out_color %f\n",out_color[ch * H * W + pix_id]);
        }
        //if(isnan(out_color[pix_id]))printf("blkx%d,blky%d,thx%d,thy%d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y);
        if(blockIdx.x==15&&blockIdx.y==15&&threadIdx.x==15&&threadIdx.y==3)printf("C%f\n",C[0]);
        //if(blockIdx.x==37&&blockIdx.y==36&&threadIdx.x==15&&threadIdx.y==3)printf("T%f\n",T);
        

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
        
        
        
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
    //if(pix_id==249928)printf("final_Tforward%f\n",final_T[pix_id+2 * H * W]);

}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
render1CUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
    //if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0)printf("success1 ");
    //printf("render1Cuda\n");
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};
    
    
    
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
        
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
        
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
		}
		block.sync();
        
        
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
    //const int gs2tri_sz,
    const int* trian,
    //const int trian_sz,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* colors,
	const float* transMats,
	const float* depths,
	const float4* normal_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{
    //std::cout<<"trian1"<<trian[0]<<" "<<trian[1]<<" "<<trian[23168]<<" "<<trian[23169]<<std::endl;
    std::cout<<"rua"<<std::endl;
    
    
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
        trian,
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
    
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec2* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
