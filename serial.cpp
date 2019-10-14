#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#define MAX_CITIES 1002 
#define MAX_ANTS 501
#define Q 80
#define ALPHA 0.5
#define BETA 0.8 
#define RHO 0.5 

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
	for(int i=0;i<MAX_CITIES;i++)
	{
		for(int j=0;j<MAX_CITIES;j++)
		{
			dist[i][j]=0.0f;
			pheromone[i][j]=(1.0f/n);
			Delta_Pheromones[i][j] = 0.0f;
			if(i!=j)
			{
				dist[i][j]=sqrt(pow(fabs(city[i].x-city[j].x),2)+pow(fabs(city[i].y-city[j].y),2));
			}
		}	
	}
}
void initializeTour(){
	
	s = 0;


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
		for(int k = 0; k < MAX_ANTS ; k++){
			j = nextCity(k, n);
				
			ant[k].nextCity = j;
			ant[k].visited[j]=1;
			ant[k].tour[s] = j;			
			ant[k].L+=dist[ant[k].curCity][j];
			
			ant[k].curCity = j;
		}
	}
}
void endTour(){
	
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
			Delta_Pheromones[first][second] += Q/ant[k].L;
		}
	}
}
int updatePheromone(){
	
	for(int i =0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
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
	t += MAX_ANTS;
	NC += 1;
}
void emptyTour(){
	cout<<"Clearing Tour"<<endl;
	for(int k = 0;k<MAX_ANTS;k++){
		for(int i = 0; i < MAX_CITIES;i++){
			ant[k].tour[i] = 0;
			ant[k].visited[i] = 0;
		}
	}
}

int main(int argc, char *argv[])
{	

	clock_t start = clock();
	if (argc > 1){
		cout << "Accessing file"<< argv[1]<<endl;
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
	initialize();
	int MAX_TIME = 30;
	for(;;)
	{   
		srand(time(NULL));
        initializeTour();
		tourConstruction();
		endTour();
		updatePheromone();
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


	cout<<"\n Best tour = "<<best<<endl<<endl<<endl;

	clock_t last = clock();

	cout<< double(last - start) / CLOCKS_PER_SEC <<endl;
	return 0;
}
