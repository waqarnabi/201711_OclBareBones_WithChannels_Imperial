//-----------------
//Problem Size
//-----------------
//Make sure it matches size in main.cpp
#define SIZE      10

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

//CPU/GPU run is ALWAYS run in DEBUG mode
#if TAR_CPU_OR_GPU
  #define DEBUG
#endif


// -------------------------------
// AOCL specific
// -------------------------------
#if TARGET==AOCL
#pragma OPENCL EXTENSION cl_altera_channels : enable
#endif

// ---------------------------------
// Channels (AOCL), PIPES (Intel-CPU)
// ----------------------------------
#if TARGET==AOCL
  channel int ch0;
  channel int ch1;
#elif TARGET==CPU
  //for CPU targets, pipes are declared on host and passed
  //as arguments to the kernels
#else
  #error "Invalid TARGET defined"
#endif


//------------------------------------------
// Read memory kernel
//------------------------------------------
__kernel void kernelInput( __global int * restrict aIn
#if TAR_CPU_OR_GPU  
                        , write_only pipe int ch0
#endif                        
                        ) {
  int data;
  for(int i=0; i<SIZE; i++) {
    //read from global memory
    data = aIn[i];
    
    //write to channel
#if TARGET==AOCL
    write_channel_altera(ch0, data);
#elif TAR_CPU_OR_GPU
    //printf("CPU KERNEL run\n");
    write_pipe(ch0, &data);
    #ifdef DEBUG    
      printf("DEV::kernelInput: i = %d, written to ch0: %d\n", i, data);
    #endif
#endif           
  }
}//() 

//----------------------
//Compute kernel
//----------------------
__kernel void kernelCompute ( 
#if TAR_CPU_OR_GPU  
                         read_only  pipe int ch0
                        ,write_only pipe int ch1
#endif                        
                            ) {

  //3-point window                              
  int window[3];
  int dataIn, dataOut;
  int ii, io; //index of input and index of output


#ifdef DEADLOCK
  //XX WRONG XX loop size (simply looping to size of array)
  //results in 1 less output generated from this kernel, 
  //as io (output counter), which trails ii (input counter) but 1, 
  //will not have the opportunity to count to the end of the array 
  //(will miss last element in this example)
  for(int ii=0; ii < SIZE; ii++) {  
#else       
    //printf("Doing the normal (NO DEADLOCK) run\n");
  //$$ CORRECT $$ loop size is: size of array + MAX-pos-offset (where max-pos-off = 1)
  for(int ii=0; ii < (SIZE+1); ii++) {
#endif    
    //the output index trails behind the input by 1 
    //(i.e. MAX_POSITIVE_OFFSET)
    io = ii-1;
    
    //read channel only when ii < SIZE (i.e., dont read for the last two iteration)
    if(ii < SIZE) {
#if TARGET==AOCL
      dataIn = read_channel_altera(ch0);
#elif TAR_CPU_OR_GPU
      read_pipe(ch0, &dataIn);
      #ifdef DEBUG    
        printf("DEV::kernelCompute: ii = %d, read from ch0: %d\n", ii, dataIn);
      #endif
#endif    
    }
    
    //fill up the window buffer; shifts right //
#if TARGET==AOCL
    #pragma unroll 
#endif 
    for (int j = 2; j >= 1 ; --j) 
      window[j] =     window[j - 1];

    if(ii < SIZE)
      window[0] = dataIn;

       
    //compute and write only when ii >=2 (or io >= 0) which is when we have all the window elements
    if(io >= 0) {
      //boundary
      if((io==0) || (io==SIZE-1))
        dataOut = window[1];
      //3-point averaging window
      else
        dataOut = (window[0] + window[1] + window[2]) / 3;      
      
      //write to channel
#if TARGET==AOCL
      write_channel_altera(ch1, dataOut);
#elif TAR_CPU_OR_GPU
      write_pipe(ch1, &dataOut);
      #ifdef DEBUG    
        printf("DEV::kernelCompute: io = %d, written to ch1: %d\n", io, dataOut);
      #endif
#endif  
    }                          
  }
}//()


//------------------------------------------
// Write memory kernel
//------------------------------------------
kernel void kernelOutput ( __global int * restrict aOut
#if TAR_CPU_OR_GPU  
                          ,read_only pipe int ch1
#endif                          
                          ) {
  int data;
  for (int i=0; i < SIZE; i++) {
    //read from channel
#if TARGET==AOCL
    data = read_channel_altera(ch1);
#elif TAR_CPU_OR_GPU
    while(get_pipe_num_packets(ch1)==0);
    read_pipe(ch1, &data);
    #ifdef DEBUG    
      printf("DEV::kernelOutput: i = %d, read from ch1: %d\n", i, data);
    #endif  
#endif   
    
    //write to global mem
    aOut[i] = data;
  }
}//()
