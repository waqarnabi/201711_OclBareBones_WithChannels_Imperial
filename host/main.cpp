//-----------------
//Define Target
//-----------------
#define AOCL    1
#define CPU     2
#define GPU     3

#ifndef TARGET
  #define TARGET CPU
#endif

#define TAR_CPU_OR_GPU (TARGET==CPU) || (TARGET==GPU)
//-----------------
//Preprocessing
//-----------------

//let me use deprected commands (for clCreateCommandQueue)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define CHECK(X) assert(CL_SUCCESS == (X))

//-----------------
//Problem Size
//-----------------
//Make sure it matches size in kernels.cl

#define SIZE      10
  //size of input array
#define MAXINPUT  10
  //randomly populated elements in the input array will not exceed this value

//--------------------------------
//CHANNELS, # OF KERNELS
//--------------------------------
//#define CHANNELS

#ifdef CHANNELS
  #define NKERNELS    3
  #define K_INPUT     0
  #define K_COMPUTE   1
  #define K_OUTPUT    2
  #define NAME_K0 "kernelInput"
  #define NAME_K1 "kernelCompute"
  #define NAME_K2 "kernelOutput"
  const char* kernel_names[NKERNELS] = 
     {
        NAME_K0
       ,NAME_K1
       ,NAME_K2
     };
#else
  #define NKERNELS    1
  #define K_COMPUTE   0
  #define NAME_K0     "kernelCompute"
  const char* kernel_names[NKERNELS] = 
     {
        NAME_K0
     };
#endif




//-------------------------------------------------
//signatures 
//-------------------------------------------------
void init(int a[]);

void kernelHostVersion(  int  a_in[]
                        , int a_out[]) ;
                
int post  ( FILE *fp
          , FILE *fperor
          , int *aIn
          , int *aOutHost
          , int *aOutDev
          );

void notify_print( const char* errinfo, const void* private_info, size_t cb, void *user_data );

unsigned char *load_file(const char* filename,size_t*size_ret);

int generate_random_int()
{
    return (rand());
}
          
//-------------------------------------------------
//globals
//-------------------------------------------------
const unsigned char *binary = 0;
cl_platform_id    platform;
cl_device_id      device;
cl_context        context;
cl_program        my_program;

//multiple kernels (possibly over  multiple devices) required multiple command queues
cl_kernel         kernels[NKERNELS];
#if TAR_CPU_OR_GPU
cl_command_queue  commands[1]; 
#else
cl_command_queue  commands[NKERNELS]; 
#endif


//-------------------------------------------------
//main()
//-------------------------------------------------
int main(int argc, char**argv) 
{
  #ifdef CHANNELS
    printf("HST::Executing **channelized** version\n");
  #else
    printf("HST::Executing **non-channelized** version\n");
  #endif
//variables
  int aIn[SIZE];     
  int aOutHostRun[SIZE];  
  int aOutOclRun[SIZE]; 

  //initialize output file
  FILE *fp, *fperror;
  fp      = fopen("out.dat", "w");
  fperror = fopen("error.log", "w");

  //initialize variables
  init(aIn);

//-------------------------------------------------
//hostOnly-run
//-------------------------------------------------
  kernelHostVersion ( aIn
                    , aOutHostRun
                    );  

//-------------------------------------------------
//opencl run
//-------------------------------------------------

  //ocl device memory pointers
  //---------------------
  cl_mem aInDev  = 0;
  cl_mem aOutDev = 0;
  
  //other variables
  //---------------------
  cl_int status = 0;
  int    num_errs = 0;
  int    i;
  
  cl_uint num_platforms = 0;
  
  //platform,device, context, command queue
  //---------------------------------------
  CHECK ( clGetPlatformIDs  (1, &platform, NULL) );       
  //get platform vendor
  char cl_platform_vendor[1001];
  char cl_platform_version[51];
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor, NULL) );
  CHECK (clGetPlatformInfo ( platform, CL_PLATFORM_VERSION, 50, (void *)cl_platform_version, NULL) );
  printf("HST::CL_PLATFORM_VENDOR:\t%s\t:: Version: %s\n",cl_platform_vendor, cl_platform_version);   

   
#if TARGET==AOCL
  printf("HST::Getting AOCL FPGA target device\n");
  CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,0) );
#elif TARGET==CPU
  printf("HST::Getting CPU target device\n");
  CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_CPU        ,1,&device,0) );
#elif TARGET==GPU
  printf("HST::Getting GPU target device\n");
  CHECK( clGetDeviceIDs  (platform,CL_DEVICE_TYPE_GPU        ,1,&device,0) );
#else
  #error "Unknown TARGET specificed."
#endif

  context = clCreateContext( 0, 1, &device, notify_print, 0, &status );
  CHECK( status );

  //single command queue for CPU (see README for explanation)
  #if TAR_CPU_OR_GPU
      commands[0] = clCreateCommandQueue( context, device, 0, &status );
  #else      
    //Create separate queue for each kernel, even if on same device  
    for (i=0; i<NKERNELS; i++) {
        commands[i] = clCreateCommandQueue( context, device, 0, &status );
      CHECK( status );
    }
  #endif

  //pipes
  //---------------------------------------
  #if TARGET==AOCL
    //AOCL creates channels (pipes)  in the kernel file global scope
  #else
    cl_mem ch0 = 0;
    cl_mem ch1 = 0;
  
    //Create and initialize memory objects
    const int PSIZE=SIZE;
    ch0 = clCreatePipe(context,CL_MEM_READ_WRITE,sizeof(int),PSIZE,NULL,&status);
    CHECK( status );
    ch1 = clCreatePipe(context,CL_MEM_READ_WRITE,sizeof(int),PSIZE,NULL,&status);
    CHECK( status );
  
  printf("HST::Created pipes ch0 and ch1 in host scope, to be passed to kernels\n");
  #endif


  //load kernel
  //------------
//For FPGA target, the kernels are compiled offline (via aoc compiler), and binary is loaded   
#if TARGET==AOCL   
  cl_int               bin_status = 0;
  size_t               bin_len = 0;
  const unsigned char* my_binary;
  size_t               my_binary_len = 0;
  #ifdef CHANNELS  
    const char *         aocx_name = "kernels_channels.aocx";
  #else
    const char *         aocx_name = "kernels_noChannels.aocx";
  #endif    
  printf("HST::Loading kernel binary %s ...\n", aocx_name);
  
  my_binary = load_file(aocx_name, &my_binary_len); 

  if ((my_binary == 0) || (my_binary_len == 0))
  { 
    printf("HST::Error: unable to read %s into memory or the file was not found!\n", aocx_name);
    exit(-1);
  }

  my_program = clCreateProgramWithBinary(context,1,&device,&my_binary_len,&my_binary,&bin_status,&status);
  CHECK(status);
//For CPU/GPU targets, the kernels are compiled at runtime
#else
  #ifdef CHANNELS
    const char *clsourceFile  = "../device/kernels_channels.cl";
  #else
    const char *clsourceFile  = "../device/kernels_noChannels.cl";
  #endif


  printf("HST::Reading kernel source file: %s\n", clsourceFile);

  // get size of kernel source
  FILE*  kfileHandle;
  size_t kfileSize;
  kfileHandle = fopen(clsourceFile, "r");
  fseek(kfileHandle, 0, SEEK_END);
  kfileSize = ftell(kfileHandle);
  rewind(kfileHandle);

  // read kernel source into buffer
  char *clsource = (char*) malloc(kfileSize + 1);
  clsource[kfileSize] = '\0';
  fread(clsource, sizeof(char), kfileSize, kfileHandle);
  fclose(kfileHandle);

  my_program = clCreateProgramWithSource ( context
                                      , 1
                                      , (const char **) &clsource 
                                      , NULL
                                      , &status
                                      );
  CHECK (status);
  printf("HST::Created program with source: %s\n", clsourceFile);
#endif  
   
  //for CPU/GPU options, we are compiling kernel at runtime, so
  //we need to pass it pre-processor flags here
  #if TARGET==CPU
    #ifdef DEADLOCK
      const char *preProcFlags="-DTARGET=CPU -DDEADLOCK -cl-std=CL2.0";
    #else 
      const char *preProcFlags="-DTARGET=CPU -cl-std=CL2.0";
   #endif
  #elif TARGET==GPU
    #ifdef DEADLOCK
      const char *preProcFlags="-DTARGET=GPU -DDEADLOCK -cl-std=CL2.0";
    #else 
      const char *preProcFlags="-DTARGET=GPU -cl-std=CL2.0";
   #endif
  #else  
      const char *preProcFlags="";
  #endif  


  printf("HST::Building program\n");
  CHECK( clBuildProgram(my_program,1,&device,preProcFlags,0,0) );
  /* 
  //If you get build error and want to see build messages, uncomment this block
  
  clBuildProgram(my_program,1,&device,preProcFlags,0,0);
  size_t len;
  clGetProgramBuildInfo(my_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char buffer[len];
  clGetProgramBuildInfo(my_program, device, CL_PROGRAM_BUILD_LOG, len*sizeof(char), buffer, &len);
  printf("HST::%s\n", buffer);
  clGetProgramBuildInfo(my_program, device, CL_PROGRAM_BUILD_STATUS, len*sizeof(char), buffer, &len);
  printf("HST::%s\n", buffer);
  clGetProgramBuildInfo(my_program, device, CL_PROGRAM_BUILD_OPTIONS, len*sizeof(char), buffer, &len);
  printf("HST::%s\n", buffer);
  */
  
  printf("HST::Creating kernel(s)\n");
  //create kernel(s)
  for (i=0; i<NKERNELS; i++) {  
   kernels[i] = clCreateKernel(my_program, kernel_names[i], &status);
   CHECK(status);
  }    

  //cl buffers
  //---------------------
  printf("HST::Creating cl (device) buffers\n");
  aInDev  = clCreateBuffer(context,CL_MEM_READ_WRITE,SIZE*sizeof(int),0,&status); CHECK(status);
  aOutDev = clCreateBuffer(context,CL_MEM_READ_WRITE,SIZE*sizeof(int),0,&status); CHECK(status);

  //Prepare Kernel, Args
  //---------------------
  printf("HST::Preparing kernels\n");   
  int cl_wi=0;
  
  //write initial data to buffer on device (0th command queue for 0th kernel)
  CHECK( clEnqueueWriteBuffer(commands[0],aInDev,0,0,SIZE*sizeof(int),aIn,0,0,0) );
  CHECK( clFinish(commands[0]) );
  //£
  //for (int i=0; i<NKERNELS; i++)
  //   CHECK( clFinish(commands[i]) );
  
  size_t dims[3] = {0, 0, 0};    
  //Currently I am working with Single work-instance kernels, so that
  //looping in the kernel can be explored, which is more suitable for FPGA targets
  dims[0] = 1 ;
  
  #ifdef CHANNELS
    CHECK( clSetKernelArg(kernels[K_INPUT]  ,0,sizeof(cl_mem),&aInDev)  );
    CHECK( clSetKernelArg(kernels[K_OUTPUT] ,0,sizeof(cl_mem),&aOutDev)  );
    //if cpu or gpu target (and channelized) then pipes need to passed too
    #if TAR_CPU_OR_GPU
      CHECK( clSetKernelArg(kernels[K_INPUT]  ,1,sizeof(cl_mem),(void *) &ch0 ));
      CHECK( clSetKernelArg(kernels[K_COMPUTE],0,sizeof(cl_mem),(void *) &ch0 ));
      CHECK( clSetKernelArg(kernels[K_COMPUTE],1,sizeof(cl_mem),(void *) &ch1 ));
      CHECK( clSetKernelArg(kernels[K_OUTPUT] ,1,sizeof(cl_mem),(void *) &ch1 ));
    #endif
  #else
    CHECK( clSetKernelArg(kernels[K_COMPUTE],0,sizeof(cl_mem),&aInDev)  );
    CHECK( clSetKernelArg(kernels[K_COMPUTE],1,sizeof(cl_mem),&aOutDev)  );
  #endif
 
  //Launch Kernel
  //-------------
  #ifdef DEADLOCK
    printf("HST::Doing the **DEADLOCK** run\n");
  #endif    
  printf("HST::Enqueueing kernel with global size %d\n",(int)dims[0]);  
  for (int i=0; i<NKERNELS; i++) {
    #if TAR_CPU_OR_GPU    
      CHECK( clEnqueueNDRangeKernel(commands[0],kernels[i],1,0,dims,0,0,0,0));
    #else
      CHECK( clEnqueueNDRangeKernel(commands[i],kernels[i],1,0,dims,0,0,0,0));
    #endif
  }
  
  #if TAR_CPU_OR_GPU
    CHECK( clFinish(commands[0]) );
  #else
    for (int i=0; i<NKERNELS; i++)
      CHECK( clFinish(commands[i]) );
  #endif

  //Read results
  //---------------------
  printf("HST::Reading results to host buffers...\n");
  #ifdef CHANNELS
    #if TAR_CPU_OR_GPU
      CHECK( clEnqueueReadBuffer(commands[0],aOutDev,1,0,SIZE*sizeof(int),aOutOclRun,0,0,0) );
    #else
      CHECK( clEnqueueReadBuffer(commands[K_OUTPUT],aOutDev,1,0,SIZE*sizeof(int),aOutOclRun,0,0,0) );
    #endif
  #else
    CHECK( clEnqueueReadBuffer(commands[K_COMPUTE],aOutDev,1,0,SIZE*sizeof(int),aOutOclRun,0,0,0) );
  #endif
   
  //Compare results
  //---------------------
  printf("HST::Checking results...\n");   
  post (fp, fperror, aIn, aOutHostRun, aOutOclRun);

  //Post-processing
  //---------------------
  clReleaseMemObject(aInDev);
  clReleaseMemObject(aOutDev);
  //clReleaseKernel(kernel);
  clReleaseProgram(my_program);
  clReleaseContext(context);
   
  return 0;
}

//-------------------------------------------------
//init()
//-------------------------------------------------
void init(  int a[SIZE]
         ){      
    for(int i=0; i<SIZE; i++) 
        a[i] = rand() % MAXINPUT;
        //a[i] = i+1;
}

//--------------------------------------
//- kernel_hostVersion (baseline)
//--------------------------------------
void kernelHostVersion ( int aIn[SIZE]
                       , int aOut[SIZE]) {
  printf("HST::Running host-only application, creating golden reference\n");
  for(int i=0; i<SIZE; i++) {
    //closed boundary
    if((i==0) || (i==SIZE-1))
      aOut[i] = aIn[i];
    //3-point averaging window
    else
      aOut[i] = (aIn[i-1] + aIn[i] + aIn[i+1]) / 3;            
  }
  printf("HST::Host-only run complete\n\n");
}//()


//--------------------------
//-Writing the arrays to file
//--------------------------
int post (  FILE *fp
          , FILE *fperror
          , int aIn[SIZE]
          , int aOutHost[SIZE]
          , int aOutDev[SIZE]
          ) {

    int num_errs = 0;
    int testPass = 1;
    
    //linear display with indices, and also compare results
    fprintf(fp, "-------------------------------------------------------------\n");
    fprintf(fp, "       i ::  aIn,   aOutHost , aOutDev\n");
    fprintf(fp, "-------------------------------------------------------------\n");

    for(int i=0; i<SIZE; i++) {
        //print to file
        fprintf (fp, "\t%5d\t::\t%10d\t%10d\t%10d\n"
                 , i,  aIn[i],  aOutHost[i], aOutDev[i]);
        //compare hostRUn <--> OpenCLrun

        if (aOutHost[i] != aOutDev[i]) {
          num_errs++;
          fprintf (fperror, "ERROR at i=%d, Expected = %d, Computed = %d\n", i, aOutHost[i], aOutDev[i]);
          testPass=0;
       }
    }//for
    if (testPass==1)
     printf("HST::$$$ Test PASSED! :) $$$\n");
    else
     printf("HST::XXX Test FAILED! :( XXX\n");
  return num_errs;
}//()


//-------------------------------------------------
//notify_print
//-------------------------------------------------

void notify_print( const char* errinfo, const void* private_info, size_t cb, void *user_data )
{
   private_info = private_info;
   cb = cb;
   user_data = user_data;
   printf("HST::Error: %s\n", errinfo);
}


//-------------------------------------------------
//load_file
//-------------------------------------------------
unsigned char *load_file(const char* filename,size_t*size_ret)
{
   FILE* fp;
   int len;
   const size_t CHUNK_SIZE = 1000000;
   unsigned char *result;
   size_t r = 0;
   size_t w = 0;
   fp = fopen(filename,"rb");
   if ( !fp ) return 0;
   // Obtain file size.
   fseek(fp, 0, SEEK_END);
   len = ftell(fp);
   // Go to the beginning.
   fseek(fp, 0, SEEK_SET);
   // Allocate memory for the file data.
   result = (unsigned char*)malloc(len+CHUNK_SIZE);
   if ( !result )
   {
     fclose(fp);
     return 0;
   }
   // Read file.
   while ( 0 < (r=fread(result+w,1,CHUNK_SIZE,fp) ) )
   {
     w+=r;
   }
   fclose(fp);
   *size_ret = w;
   return result;
}
