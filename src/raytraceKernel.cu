// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
#include <set>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define MAXDEPTH 10 // max raytrace depth

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
  ray r;
  r.origin = eye;

	// values for computing ray direction
  float phi = glm::radians(fov.y);
	float theta = glm::radians(fov.x);
	glm::vec3 A = glm::cross(view, up);
	glm::vec3 B = glm::cross(A, view);
	glm::vec3 M = eye + view;
	glm::vec3 V = B/glm::length(B) * glm::length(view) * tan(phi);
	glm::vec3 H = A/glm::length(A) * glm::length(view) * tan(theta);

	glm::vec3 P = M + (2*x/(resolution.x-1)-1) * H + (2*(1-y/(resolution.y-1))-1) * V;
	r.direction = glm::normalize(P-eye);

  return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytraceRay(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
														staticGeom* geoms, int numberOfGeoms, material* materials, int* lightIds, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
		if (rayDepth > MAXDEPTH) {
			colors[index] = glm::vec3(0, 0, 0);
		}
		else {
			ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);

			// find the closest intersected geometry
			int minIdx = -1;
			float minDist = FLT_MAX;
			glm::vec3 minIntersection, minNormal;
			for (int i=0; i<numberOfGeoms; ++i) {
				glm::vec3 intersection, normal;
				float dist = geomIntersectionTest(geoms[i], r, intersection, normal);
				if (dist > THRESHOLD && dist < minDist) {
					minIdx = i;
					minDist = dist;
					minIntersection = intersection;
					minNormal = normal;
				}
			}

			if (minIdx != -1) {
				material mtl = materials[geoms[minIdx].materialid]; // does caching make it faster?
				if (mtl.emittance > THRESHOLD) { // light
					colors[index] = glm::clamp(mtl.color * mtl.emittance, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
				}
				else {
					if (true) {
					//if (mtl.hasReflective < THRESHOLD && mtl.hasRefractive < THRESHOLD) { // diffuse surface
						glm::vec3 totalLightColor(0, 0, 0);
						for (int i=0; i<numberOfLights; ++i) {
							staticGeom light = geoms[lightIds[i]];
							glm::vec3 pointOnLight = light.translation;
							//glm::vec3 pointOnLight = getRandomPointOnGeom(light, index * time);
							float distToLight = glm::distance(minIntersection, pointOnLight);
							glm::vec3 L = glm::normalize(pointOnLight - minIntersection); // direction from point to light
							//ray shadowFeeler;
							//shadowFeeler.origin = minIntersection;
							//shadowFeeler.direction = L;

							//// find out if the ray intersects other objects
							//bool shadow = false;
							//for (int j=0; j<numberOfGeoms; ++j) {
							//	if (j != lightIds[i]) {
							//		glm::vec3 intersection, normal;
							//		float dist = geomIntersectionTest(geoms[i], shadowFeeler, intersection, normal);
							//		if (abs(dist+1) > THRESHOLD && dist < distToLight) {
							//			shadow = true;
							//			break;
							//		}
							//	}
							//}

							if (true) {
								material lightMtl = materials[light.materialid];
								glm::vec3 lightColor = glm::clamp(lightMtl.color * lightMtl.emittance, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
								lightColor *= glm::clamp(glm::dot(L, minNormal), 0.0f, 1.0f);
								totalLightColor += lightColor;
								if (totalLightColor[0] > 1 && totalLightColor[1] > 1 && totalLightColor[2] > 1) {
									break;
								}
							}
						}
						totalLightColor = glm::clamp(totalLightColor, glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));
						colors[index] = glm::vec3(mtl.color[0] * totalLightColor[0], mtl.color[1] * totalLightColor[1], mtl.color[2] * totalLightColor[2]);
					}
				}
			}
			else {
				colors[index] = glm::vec3(0, 0, 0);
			}
		}
  }
}

//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  
  int traceDepth = 1; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  // keep track of the IDs of light materials
	std::set<int> lightMtlIds;
	for (int i=0; i<numberOfMaterials; ++i) {
		if (materials[i].emittance > 0) {
			lightMtlIds.insert(i);
		}
	}
	
	//package geometry and materials and sent to GPU
	// keep track of the IDs of light geometries
  staticGeom* geomList = new staticGeom[numberOfGeoms];
	std::vector<int> lightIds;
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
		if (lightMtlIds.find(newStaticGeom.materialid) != lightMtlIds.end()) {
			lightIds.push_back(i);
		}
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
	int* lightIdList = new int[lightIds.size()];
	for (int i=0; i<lightIds.size(); i++) {
		lightIdList[i] = lightIds[i];
	}
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

	// copy materials to CUDA
	material* cudamtls = NULL;
	cudaMalloc((void**)&cudamtls, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamtls, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

	// copy lights to CUDA
	int* cudalights = NULL;
	cudaMalloc((void**)&cudalights, lightIds.size()*sizeof(int));
	cudaMemcpy( cudalights, lightIdList, lightIds.size()*sizeof(int), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  //kernel launches
  raytraceRay<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage,
		cudageoms, numberOfGeoms, cudamtls, cudalights, lightIds.size());

  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
	cudaFree( cudamtls );
	cudaFree( cudalights );
  delete geomList;
	delete lightIdList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
