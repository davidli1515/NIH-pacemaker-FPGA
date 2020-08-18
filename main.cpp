#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream> // To use ifstream
#include "unet.h"
#include <ap_int.h>
using namespace std;

//
//#define dataw 32
//#define FLOAT 1
//
//#if FLOAT==1
//	typedef float data_type;
//#else
//	typedef ap_fixed<8,3> data_type;
//#endif
//
//struct data_pack{
//	data_type data0;
//	data_type data1;
//};
//
//struct dma_data{
//	data_pack data;
//};
//const int TmBuff1=2, TnBuff1=2,Tr1=4,Tc1=4,Tm1=2,Tn1=2,Tk1=7,Tri1=10,Tci1=10;
//const int M1 = 48, N1=2,C1=16,H1=22, K1=7, S1=1;




int main(){
    ifstream inputFile;
    // data_type * flatten_weight= new data_type[M1*N1*K1*K1]( include)
    data_type * flatten_weight=(data_type *)malloc((M1*N1*K1*K1+M1)*sizeof(data_type));
    data_type * flatten_input=(data_type *)malloc(N1*C1*C1*sizeof(data_type));
    data_type * flatten_output=(data_type *)malloc(M1*C1*C1*sizeof(data_type));
    data_type * golden_output=(data_type *)malloc(M1*C1*C1*sizeof(data_type));
    inputFile.open("test_weight.inc");

   for(unsigned int i=0; i<M1;i++)
       for(unsigned int j=0; j<N1;j++)
           for(unsigned int k=0; k<K1; k++)
               for(unsigned int l=0; l<K1; l++)
               {
            	   inputFile>>flatten_weight[i*N1*K1*K1+j*K1*K1+k*K1+l];
               }
    for(unsigned int i=0; i<M1;i++)
        inputFile>>flatten_weight[M1*N1*K1*K1+i];
    inputFile.close();
    
    
    
    inputFile.open("test_x.inc");
   
           
   for(unsigned int i=0; i<N1; i++)
	   for(unsigned int j=0; j<C1; j++)
		   for(unsigned int k=0; k<C1; k++){
			   inputFile>>flatten_input[i*C1*C1+j*C1+k];
		   }
    inputFile.close();
    
    inputFile.open("test_output.inc");
    for(unsigned int i=0; i<M1; i++)
	   for(unsigned int j=0; j<C1; j++)
		   for(unsigned int k=0; k<C1; k++){
			   inputFile>>golden_output[i*C1*C1+j*C1+k];
		   } 
    inputFile.close();
   // for(unsigned int i=0; i<M1; i++)
	   // for(unsigned int j=0; j<C1; j++)
		   // for(unsigned int k=0; k<C1; k++){
			   // flatten_output[i*C1*C1+j*C1+k]=(data_type)output[i][j][k];
		   // }
           
    dma_data * packed_weight=(dma_data *)malloc((M1*N1*K1*K1+M1)/2*sizeof(dma_data));
    dma_data * packed_input=(dma_data *)malloc(N1*C1*C1/2*sizeof(dma_data));
    dma_data * packed_output=(dma_data *)malloc(M1*C1*C1/2*sizeof(dma_data));
    dma_data * packed_input2=(dma_data *)malloc(N2*C2*C2/2*sizeof(dma_data));
    for(unsigned int i=0; i<M1;i+=2)
       for(unsigned int j=0; j<N1;j++)
           for(unsigned int k=0; k<K1; k++)
               for(unsigned int l=0; l<K1; l++)
               {
                   packed_weight[i/2*N1*K1*K1+j*K1*K1+k*K1+l].data.data0=flatten_weight[i*N1*K1*K1+j*K1*K1+k*K1+l];
                   packed_weight[i/2*N1*K1*K1+j*K1*K1+k*K1+l].data.data1=flatten_weight[(i+1)*N1*K1*K1+j*K1*K1+k*K1+l];
               }
    for(unsigned int i=0; i<M1;i+=2){
        packed_weight[M1*N1*K1*K1/2+i/2].data.data0=flatten_weight[M1*N1*K1*K1+i];
        packed_weight[M1*N1*K1*K1/2+i/2].data.data1=flatten_weight[M1*N1*K1*K1+i+1];
    }
    
    for(unsigned int i=0; i<N1; i+=2)
	   for(unsigned int j=0; j<C1; j++)
		   for(unsigned int k=0; k<C1; k++){
			   packed_input[i/2*C1*C1+j*C1+k].data.data0=flatten_input[i*C1*C1+j*C1+k];
               packed_input[i/2*C1*C1+j*C1+k].data.data1=flatten_input[(i+1)*C1*C1+j*C1+k];
		   }
    
    unet_top (
        packed_weight,
        packed_input,
        packed_output,
        // dma_data* weight2,
        packed_input2,
        // dma_data* output_core2,
        // dma_data* weight3,
        // dma_data* feature3,
        // dma_data* output_core3,
        // dma_data* weight4,
        // dma_data* feature4,
        // dma_data* output_core4,
        // dma_data* weight5,
        // dma_data* feature5,
        // dma_data* output_core5,
        // dma_data* weight6,
        // dma_data* feature6,
        // dma_data* output_core6,
        // dma_data* weight7,
        // dma_data* feature7,
        // dma_data* output_core7,
        // dma_data* weight8,
        // dma_data* feature8,
        // dma_data* output_core8,
        //dma_data* weight9,
        //dma_data* feature9,
        //dma_data* output_core9,
        //dma_data* weight10,
        //dma_data* feature10,
        //dma_data* output_core10,
        //dma_data* weight11,
        //dma_data* feature11,
        //dma_data* output_core11,
        //dma_data* weight12,
        //dma_data* feature12,
        //dma_data* output_core12,
        //dma_data* weight13,
        //dma_data* feature13,
        //dma_data* output_core13,
        //dma_data* weight14,
        //dma_data* feature14,
        //dma_data* output_core14,
        //dma_data* weight15,
        //dma_data* feature15,
        //dma_data* output_core15,
        //dma_data* weight16,
        //dma_data* feature16,
        //dma_data* output_core16,
        //dma_data* weight17,
        //dma_data* feature17,
        //dma_data* output_core17,
        1,
        // ap_uint<32> Base_addr1,
        // ap_uint<32>  Base_addr2,
        // ap_uint<32>  Base_addr3,
        // ap_uint<32> Base_addr4,
        // ap_uint<32>  Base_addr5,
        // ap_uint<32>  Base_addr6,
        // ap_uint<32> Base_addr7,
        // ap_uint<32>  Base_addr8,
        // ap_uint<32>  Base_addr9,
        // ap_uint<32> Base_addr10,
        // ap_uint<32>  Base_addr11,
        // ap_uint<32>  Base_addr12,
        //ap_uint<32> Base_addr13,
        //ap_uint<32>  Base_addr14,
        //ap_uint<32>  Base_addr15,
        //ap_uint<32> Base_addr16,
        //ap_uint<32>  Base_addr17,
        //ap_uint<32>  Base_addr18,
        //ap_uint<32> Base_addr19,
        //ap_uint<32>  Base_addr20,
        //ap_uint<32>  Base_addr21,
        //ap_uint<32> Base_addr22,
        //ap_uint<32>  Base_addr23,
        //ap_uint<32>  Base_addr24,
        //ap_uint<32> Base_addr25,
        //ap_uint<32>  Base_addr26,
        //ap_uint<32>  Base_addr27,
         // ap_uint<32> Base_addr28,
         // ap_uint<32>  Base_addr29,
         // ap_uint<32>  Base_addr30,
         // ap_uint<32> Base_addr31,
         // ap_uint<32>  Base_addr32,
         // ap_uint<32>  Base_addr33,
         // ap_uint<32> Base_addr34,
        0x0000,
         // ap_uint<32>  Base_addr36,
        0x0000,
        0x0000,
        0x0000
        //ap_uint<32> Base_addr40,
        //ap_uint<32>  Base_addr41,
        //ap_uint<32>  Base_addr42,
        //ap_uint<32> Base_addr43,
        //ap_uint<32>  Base_addr44,
        //ap_uint<32>  Base_addr45,
        //ap_uint<32> Base_addr46,
        //ap_uint<32>  Base_addr47,
        //ap_uint<32>  Base_addr48,
        //ap_uint<32> Base_addr49,
        //ap_uint<32>  Base_addr50,
        //ap_uint<32>  Base_addr51
    );
    
    // dma_data tmp;
    // for(unsigned int i=0; i<M1;i+=2)
       // for(unsigned int j=0; j<N1;j++)
           // for(unsigned int k=0; k<K1; k++)
               // for(unsigned int l=0; l<K1; l++)
               // {
                // tmp=packed_weight[i/2*N1*K1*K1+j*K1*K1+k*K1+l];
                // if (tmp.data.data0!= flatten_weight[i*N1*K1*K1+j*K1*K1+k*K1+l] || tmp.data.data1!= flatten_weight[(i+1)*N1*K1*K1+j*K1*K1+k*K1+l]){
                    
                    // cout<<"failed"<<' ';
                    // cout<<tmp.data.data0<<' ';
                    // cout<<tmp.data.data1<<endl;
                    // break;
                // }
               // }

    // dma_data tmp;
    // for(unsigned int i=0; i<N1; i+=2)
	   // for(unsigned int j=0; j<C1; j++)
		   // for(unsigned int k=0; k<C1; k++){
                // tmp=packed_input[i/2*C1*C1+j*C1+k];
                // if (tmp.data.data0!= flatten_input[i*C1*C1+j*C1+k] || tmp.data.data1!= flatten_input[(i+1)*C1*C1+j*C1+k]){
                    
                    // cout<<"failed"<<' ';
                    // cout<<tmp.data.data0<<' ';
                    // cout<<tmp.data.data1<<endl;
                    // break;
                // }
               // }
    dma_data tmp;           
    for(unsigned int i=0; i<M1; i+=2){
	   for(unsigned int j=0; j<C1; j++){
               cout<<'\n'<<' ';
		   for(unsigned int k=0; k<C1; k++){
			   cout<<packed_output[i/2*C1*C1+j*C1+k].data.data0<<',';
		   }
       }
	   for(unsigned int j=0; j<C1; j++){
               cout<<'\n'<<' ';
		   for(unsigned int k=0; k<C1; k++){
			   cout<<packed_output[i/2*C1*C1+j*C1+k].data.data1<<',';
		   }
       }
                      
    }
    cout<<"\n=========================================================================\n"<<' ';
    
    for(unsigned int i=0; i<M1; i+=2){
	   for(unsigned int j=0; j<C2; j++){
               cout<<'\n'<<' ';
		   for(unsigned int k=0; k<C2; k++){
			   cout<<packed_input2[i/2*C2*C2+j*C2+k].data.data0<<',';
		   }
       }
	   for(unsigned int j=0; j<C2; j++){
               cout<<'\n'<<' ';
		   for(unsigned int k=0; k<C2; k++){
			   cout<<packed_input2[i/2*C2*C2+j*C2+k].data.data1<<',';
		   }
       }
                      
    } 
    return 0;

}
