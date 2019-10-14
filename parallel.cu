#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<curand_kernel.h>
#include<curand.h>
#include<time.h>


#define MAX_CITIES 29	
#define MAX_ANTS 14		
#define Q 80
#define ALPHA 0.5
#define BETA 0.8 
#define RHO 0.5 

using namespace std;

int n=0;
int NC = 0;
int t = 0;
struct cities
{
	int x,y;
};
int s;
struct ANTS{
	
	int curCity, nextCity;
	int visited[MAX_CITIES];
	int tour[MAX_CITIES];
	float L;
};

cities city[MAX_CITIES];
float pheromone[MAX_CITIES][MAX_CITIES];
float dist[MAX_CITIES][MAX_CITIES];
ANTS ant[MAX_ANTS];
int best=9999999;
int bestIndex;
float Delta_Pheromones[MAX_CITIES][MAX_CITIES];
float numerator[MAX_CITIES][MAX_CITIES];
curandState  state[MAX_ANTS];


__global__ void initialize(float *d_dist,float *d_pheromone,float *d_Delta_Pheromones,cities *d_city,int n)
{	



	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;



	if((row<n)&&(col<n)){
	
		d_dist[col + row * n] = 0.0f;
		d_pheromone[col + row * n] = 1.0f / n;
		d_Delta_Pheromones[col + row * n] = 0.0f;
		if(row!=col)
		{
			d_dist[col + row * n]=sqrt(powf(abs(d_city[row].x-d_city[col].x),2)+powf(abs(d_city[row].y-d_city[col].y),2));
			
		}
	}



}


__global__ void setup_curand_states(curandState *state_d,int t){
	
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	curand_init(t, id, 0, &state_d[id]);
}

__device__ float generate(curandState* globalState, int ind){
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}



__global__ void initializeTour(ANTS *d_ant,int n){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n){
		int j = id;
		d_ant[id].curCity = j;
		for(int i=0;i<n;i++)
		{
			d_ant[id].visited[i]=0;
		}
		d_ant[id].visited[j] = 1;
		d_ant[id].tour[0] = j;
		d_ant[id].L = 0.0;
	}
}

__global__ void PHI_numerator(float *d_numerator, float *d_dist, float *pheromone, int n){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < n && col < n){
		int id = row * n + col;
		d_numerator[id] =  powf( pheromone[id], ALPHA) * powf( (1.0/ d_dist[id]), BETA);
	}
}

__device__ int nextCity(int k,int n,float *d_numerator,ANTS *d_ant,curandState *state_d)
{	
	int i = d_ant[k].curCity;
	int j;
	double sum=0.0;
	for(j=0;j<n;j++)
	{
		if(d_ant[k].visited[j]==0)
		{
			sum+= d_numerator[i*n+j];
		}
	}
	
	while(1)
	{
		j++;
		if(j >= n)
			j=0;
		if(d_ant[k].visited[j] == 0)
		{
			float probability = d_numerator[i*n+j]/sum;
			float random = (float)generate(state_d,i); 
			
			if(random < probability)
			{
				break;
			}
		}
	}
	
	return j;
}

__global__ void tourConstruction(ANTS *d_ant, float *d_dist, float *d_numerator,int n,curandState *state_d)
{	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n){
		for(int s=1;s<n;s++)
		{	
			int j = nextCity(id, n, d_numerator,d_ant,state_d);	
			d_ant[id].nextCity = j;
			d_ant[id].visited[j]=1;
			d_ant[id].tour[s] = j;			
			d_ant[id].L+=d_dist[d_ant[id].curCity * n + j];
			d_ant[id].curCity = j;
		}
	}
}
__global__
void endTour(float *Delta_Pheromones, ANTS *ant,float *dist, int *best, int *bestIndex){
	
	int k = threadIdx.x + blockIdx.x * blockDim.x;
	if(k < MAX_ANTS){
		ant[k].L += dist[ant[k].curCity * MAX_CITIES + ant[k].tour[0]];
		ant[k].curCity = ant[k].tour[0];
		
		int temp = *best;
		printf("This is before atomicMin %d\n", *best);
		atomicMin(best, ant[k].L);
		printf("This is after atomicMin %d\n", *best);
		if (*best!= temp){
			*bestIndex = k;
		}
		for(int i = 0; i < MAX_CITIES;i++){
			int first = ant[k].tour[i];
			int second = ant[k].tour[(i + 1) % MAX_CITIES];
			Delta_Pheromones[first * MAX_CITIES + second] += Q/ant[k].L;
		}
	}
	
}
__global__ void updatePheromone(float *d_pheromone, float *d_Delta_Pheromones, int n){

	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n){
		for(int s=0;s<n;s++){
			if(id!=s)
			{
				d_pheromone[id*n+s] *=( 1.0 - RHO);
				
				if(d_pheromone[id*n+s]<0.0)
				{
					d_pheromone[id*n+s] = (1.0/n);
				}
			}
			d_pheromone[id*n+s] += d_Delta_Pheromones[id*n+s];
			d_Delta_Pheromones[id*n+s] = 0;	
		}
	}
}
__global__ void emptyTour(ANTS *d_ant,float *d_Delta_Pheromones,int n){
	
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < n){
		
		for(int s=0;s<n;s++){		
			d_ant[id].tour[s] = 0;
			d_ant[id].visited[s] = 0;
		}	
	}
}

int main(int argc, char *argv[])


{	

	clock_t start = clock();



if (argc > 1){
		cout << "Accessing file "<< argv[1]<<endl;
	}
	else{
		cout << "Input File Name!" <<endl;
		return 1;
	}
	ifstream in;
    	in.open(argv[1]);
	in>>n;
	cout<<n<<endl;
	int num;
	for(int i=0;i<n;i++)
	{
		in>>num;	
		in>>city[i].x;
		in>>city[i].y;
		cout<<city[i].x<<" "<<city[i].y<<" "<<endl;	
	}
	
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((n - 1)/ 32 + 1, (n - 1)/ 32 + 1, 1 );
	float *d_dist,*d_pheromone,*d_Delta_Pheromones,*d_numerator;
	ANTS *d_ant;
	cities *d_city;
	curandState  *state_d;
	int *d_best, *d_bestIndex;
	cudaMalloc((void**)&d_pheromone, sizeof(float) * n * n);
	cudaMalloc((void**)&d_dist, sizeof(float) * n * n);
	cudaMalloc((void**)&d_Delta_Pheromones, sizeof(float) * n * n);
	cudaMalloc((void**)&d_ant, sizeof(ANTS) * n);
	cudaMalloc((void**)&d_city, sizeof(cities) * n);
	cudaMalloc((void**)&d_numerator, sizeof(float) * n *n);
	cudaMalloc( (void**) &state_d, sizeof(state));
	cudaMalloc((void **)&d_best, sizeof(int));
	cudaMalloc((void **)&d_bestIndex, sizeof(int));
	cudaMemcpy(d_city,city,sizeof(cities) * n,cudaMemcpyHostToDevice);
	srand(time(0));
        cudaMemcpy(d_best, &best, sizeof(int), cudaMemcpyHostToDevice);	
	int seed = rand();
	setup_curand_states <<< (n-1)/32+1,32 >>> (state_d,seed);
	initialize<<<gridDim, blockDim>>>(d_dist,d_pheromone,d_Delta_Pheromones,d_city,n);
	cudaMemcpy(dist,d_dist,sizeof(float) * n * n,cudaMemcpyDeviceToHost);
	cudaMemcpy(pheromone,d_pheromone,sizeof(float) * n * n,cudaMemcpyDeviceToHost);
	cudaMemcpy(Delta_Pheromones,d_Delta_Pheromones,sizeof(float) * n * n,cudaMemcpyDeviceToHost);
	int MAX_TIME = 20;
	for(;;)
	{		
		initializeTour<<<(n-1)/32+1,32>>>(d_ant,n);
		cudaThreadSynchronize();
		PHI_numerator<<< gridDim, blockDim>>>(d_numerator, d_dist, d_pheromone, n);
		cudaThreadSynchronize();
		tourConstruction<<<(n-1)/32+1,32>>>(d_ant,d_dist,d_numerator,n,state_d);
		cudaThreadSynchronize();
		cudaMemcpy(ant,d_ant,sizeof(ANTS) * n,cudaMemcpyDeviceToHost);
		endTour<<<(n - 1)/32 + 1, 32>>>(d_Delta_Pheromones, d_ant, d_dist, d_best, d_bestIndex);
		updatePheromone<<< (n-1)/32+1,32>>>(d_pheromone,d_Delta_Pheromones,n);
		cudaThreadSynchronize();
		t += MAX_ANTS;
		NC += 1;
		if(NC < MAX_TIME){
			emptyTour<<<(n-1)/32+1,32>>>(d_ant,d_Delta_Pheromones,n);
			cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost);
			cout<<"Best so far = "<<best<<endl;
			cudaThreadSynchronize();
		}
		else{
			break;
		}
	}
	cout<<endl;
	cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&bestIndex, d_bestIndex, sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0;i<n;i++)
	{
		cout<<ant[bestIndex].tour[i]<<" ";
	}
	cout<<endl;


	cout<<"\n Best tour = "<<best<<endl<<endl<<endl;


	clock_t last = clock();



	cout<< double(last - start) / CLOCKS_PER_SEC <<endl;
	return 0;
}