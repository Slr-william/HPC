#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NR 3
#define NC 3
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

int main (int argc, char *argv[])
{
int	numproc,nproc,numworkers, source, dest, mtype, rows, piece, residue, offset, i, j, k, rc;
double	A[NR][NC], B[NR][NC], C[NR][NC];
MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&nproc);
MPI_Comm_size(MPI_COMM_WORLD,&numproc);
if (numproc < 2 ) {
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numproc-1;

   if (nproc == MASTER)
   {
      for (i=0; i<NR; i++)
         for (j=0; j<NC; j++){
            A[i][j]= 1;//i+j;
						B[i][j]= 2;//i*j;
					}

      piece = NR/numworkers;
      residue = NR%numworkers;
      offset = 0;
      mtype = FROM_MASTER;
      for (dest=1; dest<=numworkers; dest++)
      {
         rows = (dest <= residue) ? piece+1 : piece;
         MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&A[offset][0], rows*NC, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         MPI_Send(&B, NR*NC, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
         offset = offset + rows;
      }

      mtype = FROM_WORKER;
      for (i=1; i<=numworkers; i++)
      {
         source = i;
         MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
         MPI_Recv(&C[offset][0], rows*NC, MPI_DOUBLE, source, mtype,
                  MPI_COMM_WORLD, &status);
      }

      for (i=0; i<NR; i++)
      {
         printf("\n");
         for (j=0; j<NC; j++)
            printf("%6.2f   ", C[i][j]);
      }
   }

   if (nproc > MASTER)
   {
      mtype = FROM_MASTER;
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&A, rows*NC, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&B, NC*NR, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

      for (k=0; k<NC; k++){
         for (i=0; i<rows; i++){
            C[i][k] = 0.0;
            for (j=0; j<NC; j++){
               C[i][k] = C[i][k] + A[i][j] * B[j][k];
             }
         }
       }
      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&C, rows*NC, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }
   MPI_Finalize();
}
