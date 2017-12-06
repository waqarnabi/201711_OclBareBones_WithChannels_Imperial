//Define Target
//-----------------
#define AOCL    1
#define CPU     2

//-----------------
//Problem Size
//-----------------
//Make sure it matches size in main.cpp
#define SIZE      10

//----------------------
//Compute kernel
//----------------------
__kernel void kernelCompute ( global int * restrict aIn
                            , global int * restrict aOut
                            ) {

  #if TARGET==CPU
    printf("CPU KERNEL run\n");
  #endif    
    
  for(int i=0; i<SIZE; i++) {
    //closed boundary
    if((i==0) || (i==SIZE-1))
      aOut[i] = aIn[i];
    //3-point averaging window
    else
      aOut[i] = (aIn[i-1] + aIn[i] + aIn[i+1]) / 3;      
  }                          
}//()