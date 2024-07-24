/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 保留三角光栅化
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <cmath> 
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    const int* gs2tri,
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
    //printf("dupst\n");
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
    
    
	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
                
                //printf("P%d offsets%d preoff%d \n",P,offsets[idx - 1],off);
                for(int i=0;i<6&&gs2tri[6*idx+i]!=-1;i++){
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
                //printf("off%d ",off);
                }
			}
		}
	}
}

__global__ void tri_duplicateWithKeys(
    const int* trian,
	int tri_sz,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= tri_sz)
		return;
    
    
	// Generate no key/value pair for invisible Gaussians
    
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    
    uint2 trect_min={100000,100000},trect_max={0,0};
    uint2 rect_min[3], rect_max[3];
    for(int i=0;i<3;i++){
        int pid=trian[3*idx+i];
        getRect(points_xy[pid], radii[pid], rect_min[i], rect_max[i], grid);
        trect_min.x=min(trect_min.x,rect_min[i].x);
        trect_min.y=min(trect_min.y,rect_min[i].y);
        trect_max.x=max(trect_max.x,rect_max[i].x);
        trect_max.y=max(trect_max.y,rect_max[i].y);
    }

    for(int i=trect_min.x;i<=trect_max.x;i++)
        for(int j=trect_min.y;j<=trect_max.y;j++){
            int tag=0;
            for(int k=0;k<3;k++){
                if(i>=rect_min[k].x&&i<=rect_max[k].x&&j>=rect_min[k].y&&j<=rect_max[k].y){
                tag=1;
                }
                else {
                }
            }
            if(tag){
                uint64_t key = j * grid.x + i;
                key <<= 32;
                float dep=0.0;
                for(int k=0;k<3;k++)dep+=depths[trian[3*idx+k]]/3;
                key |= *((uint32_t*)&dep);
                gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
            }
        }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
    obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	
    
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::TrianState CudaRasterizer::TrianState::fromChunk(char*& chunk, size_t P)
{
    TrianState tria;
    obtain(chunk, tria.tri_tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, tria.tri_scan_size, tria.tri_tiles_touched, tria.tri_tiles_touched, P);
    obtain(chunk, tria.tri_scanning_space, tria.tri_scan_size, 128);
    obtain(chunk, tria.tri_point_offsets, P, 128);
	return tria;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void duptile(float* t) {
    int idx = cg::this_grid().thread_rank();
    if(idx==9734)printf("trans%f\n",t[idx*9]);
    
}

__global__ void duptile1(float* dlt) {
    int idx = cg::this_grid().thread_rank();
    if(idx==16928)printf("dtrans%f\n",dlt[idx*9]);
    
}

__global__ void lengthloss(const float* mean3D,float* dL_dmean3D,int* trian,int sz) {
    int idx = cg::this_grid().thread_rank();
    if(idx>=sz)return;
    float cter[3]={0};
    for(int vt=0;vt<3;vt++){
        int pid=trian[3*idx+vt];
        for(int j=0;j<3;j++){
            cter[j]+=mean3D[3*pid+j]/3;
        }
    }
    for(int vt=0;vt<3;vt++){
        int pid=trian[3*idx+vt];
        for(int i=0;i<3;i++){
            if(mean3D[3*pid+i]>cter[i]){
                dL_dmean3D[3*pid+i]+=exp((mean3D[3*pid+i]-cter[i]-9));
            }
            else dL_dmean3D[3*pid+i]-=exp((cter[i]-mean3D[3*pid+i]-9));
            if(idx==2000)printf("exp%f\n",exp((mean3D[3*pid+i]-cter[i]-3)));
        }
    }
    
    
}

__device__ float3 vec3ToFloat3(const glm::vec3& v3) {
    float3 f3;
    f3.x = v3.x;
    f3.y = v3.y;
    f3.z = v3.z;
    return f3;
}
/*
__global__ void rotloss(const glm::vec4* rot,glm::vec4* dL_drots,const float3* mean3D,int* trian,float* dL_dR,int sz) {
    int idx = cg::this_grid().thread_rank();
    if(idx>=sz)return;
    for(int vt=0;vt<3;vt++){
        int pid=trian[3*idx+vt];
        
    }
    float3 l1=mean3D[3*idx+1]-mean3D[3*idx];
    float3 l2=mean3D[3*idx+2]-mean3D[3*idx];
    float3 axis=cross(l1,l2);
    float3 axis1=-axis;
    for(int vt=0;vt<3;vt++){
        int pid=trian[3*idx+vt];
        glm::mat3 R=quat_to_rotmat(rot[pid]);
        float cos1=sumf3(R*axis)/sqrt(R.x*R.x+R.y*R.y+R.z*R.z)/sqrt(axis.x*axis.x+axis.y*axis.y+R.z*axis.z);
        float cos2=sumf3(R*axis1)/sqrt(R.x*R.x+R.y*R.y+R.z*R.z)/sqrt(axis1.x*axis1.x+axis1.y*axis1.y+R.z*axis1.z);
    }
    
    
}
*/
__global__ void tritile(const int trian_sz, uint32_t* tri_tiles_touched, const int* trian,const float2* points_xy, int* radii,dim3 grid) {
    
    int idx = cg::this_grid().thread_rank();
    if (idx >= trian_sz) return;
    
    uint2 trect_min={100000,100000},trect_max={0,0};
    uint2 rect_min[3], rect_max[3];
    for(int i=0;i<3;i++){
        
        int pid=trian[3*idx+i];
        /*
        rect_min[i].x=1;
        rect_min[i].y=1;
        rect_max[i].x=1;
        rect_max[i].y=1;
        trect_min.x=1;
        trect_min.y=1;
        trect_max.x=1;
        trect_max.y=1;
        */
        getRect(points_xy[pid], radii[pid], rect_min[i], rect_max[i], grid);
        trect_min.x=min(trect_min.x,rect_min[i].x);
        trect_min.y=min(trect_min.y,rect_min[i].y);
        trect_max.x=max(trect_max.x,rect_max[i].x);
        trect_max.y=max(trect_max.y,rect_max[i].y);
    }
    int cnt=0;
    for(int i=trect_min.x;i<=trect_max.x;i++)
        for(int j=trect_min.y;j<=trect_max.y;j++){
            int tag=0;
            for(int k=0;k<3;k++){
                if(i>=rect_min[k].x&&i<=rect_max[k].x&&j>=rect_min[k].y&&j<=rect_max[k].y){
                tag=1;
                }
                else {
                    
                }
            }
            cnt+=tag;
        }
     tri_tiles_touched[idx]=cnt;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
    std::function<char*(size_t)> trianBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
    const int* trian,
    const int trian_sz,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_others,
	int* radii,
	bool debug)
{
    
    //std::cout<<"forward"<<std::endl;
    int* d_trian;
    CHECK_CUDA(cudaMalloc(&d_trian, 3*trian_sz*sizeof(int)),debug);
    CHECK_CUDA(cudaMemcpy(d_trian, trian, 3*trian_sz*sizeof(int), cudaMemcpyHostToDevice),debug);
    
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
    
    size_t tri_chunk_size = required<TrianState>(trian_sz);
	char* tri_chunkptr = trianBuffer(tri_chunk_size);
	TrianState trianState = TrianState::fromChunk(tri_chunkptr, trian_sz);
    
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
    
    

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);
    
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
    
    
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)
    
    //std::cout<<trianState.tri_tiles_touched<<" "<<d_trian<<" "<<std::endl;
    tritile<<<(trian_sz + 255) / 256, 256>>>(trian_sz,trianState.tri_tiles_touched,d_trian,geomState.means2D,radii,tile_grid);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error0: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error occurred");
    }
    CHECK_CUDA(, debug)
    
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    
    //
     //auto code1 = cudaDeviceSynchronize();
    //std::cout << cudaGetErrorString(code1) << std::endl;
     //CHECK_CUDA(, debug)
     
     //CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(trianState.tri_scanning_space, trianState.tri_scan_size, trianState.tri_tiles_touched, trianState.tri_point_offsets, trian_sz), debug)
    
    
	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, trianState.tri_point_offsets + trian_sz - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    //CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
    
    //std::cout<<"num_rendered"<<num_rendered<<std::endl;
    
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
    
    
    /*
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
        d_gs2tri,
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)
    */
    
    
    tri_duplicateWithKeys << <(trian_sz + 255) / 256, 256 >> > (
        d_trian,
		trian_sz,
		geomState.means2D,
		geomState.depths,
		trianState.tri_point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)
    
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : geomState.transMat;
    
    //std::cout<<"mid2"<<std::endl;
    //std::cout<<num_rendered<<std::endl;
    //assert(num_rendered==62453);
	CHECK_CUDA(FORWARD::render(
        d_trian,
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		geomState.means2D,
		feature_ptr,
		transMat_ptr,
		geomState.depths,
		geomState.normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_others), debug)
    
    CHECK_CUDA(cudaFree(d_trian),debug);
    
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
    const int* trian,
    const int trian_sz,
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* transMat_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_depths,
	float* dL_dmean2D,
	float* dL_dnormal,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dtransMat,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{

    int* d_trian;
    CHECK_CUDA(cudaMalloc(&d_trian, 3*trian_sz*sizeof(int)),debug);
    CHECK_CUDA(cudaMemcpy(d_trian, trian, 3*trian_sz*sizeof(int), cudaMemcpyHostToDevice),debug);
    
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
    
    

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
    
    //duptile<<<(P + 255) / 256, 256>>>(geomState.transMat);
    
	CHECK_CUDA(BACKWARD::render(
        d_trian,
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.normal_opacity,
		color_ptr,
		transMat_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_depths,
		dL_dtransMat,
		(float3*)dL_dmean2D,
		dL_dnormal,
		dL_dopacity,
		dL_dcolor), debug)
    
    
    //duptile1<<<(P + 255) / 256, 256>>>(dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
    
    float* dL_dRmat;
    CHECK_CUDA(cudaMalloc(&dL_dRmat, 9*P*sizeof(float)),debug);
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D, // gradient inputs
		dL_dnormal,		     // gradient inputs
		dL_dtransMat,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot,
        (glm::mat3*)dL_dRmat), debug)
    
    lengthloss<<<(trian_sz + 255) / 256, 256>>>(means3D,dL_dmean3D,d_trian,trian_sz);
    
}
