#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include <cmath>
#include<time.h>
#include<omp.h>
#include <pthread.h>
#define MAX_CITIES 48
#define MAX_ANTS 24
#define Q 80
#define ALPHA 0.5
#define BETA 0.8 
#define RHO 0.5 
#define cores 25


using namespace std;

int n=0;
int NC = 0;
int t = 0;
int s;


struct cities
{
	float x,y;
};

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
float best=(double)99999999;
int bestIndex;
float Delta_Pheromones[MAX_CITIES][MAX_CITIES];


void initialize()

{	
	
	#pragma omp parallel for
	for(int i=0;i<MAX_CITIES;i++)
	{
		for(int j=0;j<MAX_CITIES;j++)
		{
			dist[i][j]=0.0f;
			pheromone[i][j]=(1.0f/n);
			Delta_Pheromones[i][j] = 0.0f;
			if(i!=j)
			{
				 #pragma omp critical
				dist[i][j]=sqrt(pow(fabs(city[i].x-city[j].x),2)+pow(fabs(city[i].y-city[j].y),2));
			}
		}	
	}
}
void initializeTour(){
	
	s = 0;

	#pragma omp parallel for
	for(int k=0;k<MAX_ANTS;k++)
	{
		int j = rand() % MAX_CITIES;
		ant[k].curCity = j;
		for(int i=0;i<n;i++)
		{
			ant[k].visited[i]=0;
		}
		ant[k].visited[j] = 1;
		ant[k].tour[s] = j;
		ant[k].L = 0.0;
	}
}
double PHI_numerator(int i, int j)
{	
	return(( pow( pheromone[i][j], ALPHA) * pow( (1.0/ dist[i][j]), BETA)));
}

int nextCity(int k,int n)
{	
	int i = ant[k].curCity;
	int j;
	double sum=0.0;

	#pragma omp parallel for
	for(j=0;j<n;j++)
	{
		if(ant[k].visited[j]==0)
		{
			sum+= PHI_numerator(i,j);
		}
	}
	
	while(true)
	{
		j++;
		if(j >= n)
			j=0;
		if(ant[k].visited[j] == 0)
		{
			double probability = PHI_numerator(i,j)/sum;
			double random = ((double)rand()/RAND_MAX); // Calculates randomized number between 0 & 1. 
			
			if(probability>random)
			{
				break;
			}
		}
	}
	
	return j;
}

void tourConstruction()
{	
	int j;
	
	for(int s=1 ;s<n  ;s++)
	{	
		#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(cores)
		for(int k = 0; k < MAX_ANTS ; k++){
			//cout<<"im 1"<<endl;

			j = nextCity(k, n);

			//cout<<"im 2"<<endl;

				
			ant[k].nextCity = j;
			ant[k].visited[j]=1;
			ant[k].tour[s] = j;			
			ant[k].L+=dist[ant[k].curCity][j];
			
			ant[k].curCity = j;
		}
		#pragma omp barrier
	}
	//cout<<"im 3"<<endl;
}
void endTour(){
	
	#pragma omp parallel for
	for(int k = 0; k < MAX_ANTS;k++){
		ant[k].L += dist[ant[k].curCity][ant[k].tour[0]];
		ant[k].curCity = ant[k].tour[0];
		
		if(best > ant[k].L){
			best = ant[k].L;
			bestIndex = k;
		}
		for(int i = 0; i < MAX_CITIES;i++){
			int first = ant[k].tour[i];
			int second = ant[k].tour[(i + 1) % MAX_CITIES];
			#pragma omp atomic
			Delta_Pheromones[first][second] += Q/ant[k].L;
			
		}
	}
}
int updatePheromone(){
	
	for(int i =0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			#pragma omp critical
			{
			//#pragma omp flush(pheromone)//flush so that all threads see a consistent pheromone matrix
			//#pragma omp flush(Delta_Pheromones)
			if(i!=j)
			{

				pheromone[i][j] *=( 1.0 - RHO);
				
				if(pheromone[i][j]<0.0)
				{
					pheromone[i][j] = (1.0/n);
				}
			}
			pheromone[i][j] += Delta_Pheromones[i][j];
			Delta_Pheromones[i][j] = 0;

		}
			
		}
	}
	t += MAX_ANTS;
	NC += 1;
}
void emptyTour(){
	cout<<"emptyTour"<<endl;
	#pragma omp parallel for
	for(int k = 0;k<MAX_ANTS;k++){
		for(int i = 0; i < MAX_CITIES;i++){
			ant[k].tour[i] = 0;
			ant[k].visited[i] = 0;
		}
	}
}

int main(int argc, char *argv[])
{	

	
	omp_set_dynamic(0); // disable dynamic threads incase they pick smaller number of threads at runtime
	omp_set_num_threads(cores);//testing with 4 threads

	clock_t start = clock();
	if (argc > 1){
		cout << "Reading File "<< argv[1]<<endl;
	}
	else{
		cout << "Usage:progname inputFileName" <<endl;
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
	initialize();
	int MAX_TIME = 30;
	//#pragma omp parallel for
	for(;;)
	{   
		srand(time(NULL));

        initializeTour();
        //cout<<"im 1"<<endl;
		tourConstruction();
		//cout<<"im 2"<<endl;
		endTour();
		//cout<<"im 3"<<endl;
		updatePheromone();
		//cout<<"im 4"<<endl;

		if(NC < MAX_TIME){
			emptyTour();
		}
		else{
			break;
		}

	}

	cout<<endl;
	for(int i=0;i<n;i++)
	{
		cout<<ant[bestIndex].tour[i]<<" ";
	}
	cout<<endl;
	cout<<"\nSACO: Best tour = "<<best<<endl<<endl<<endl;

	clock_t last = clock();

	cout<< double(last - start) / CLOCKS_PER_SEC <<endl;
	return 0;
}

