#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>
#define SQ(x) ((x)*(x))
#define MAXITER (10000)
#define MINITER (5)

typedef struct{
	int nrow;
	int ncol;
	double *a;
} MATRIX;

void freematrix(MATRIX *mat);
void subtract(const double *x, const double *y, int n, double *sub);
void product(double *y,const double *x, double c, int n);
void matvec(double *y,const MATRIX *mat, const double *x);
double innerp(const double *x, const double *y, int n);
void cpvector(double *y, const double *x, int n);
void mkxm2(int n, double alpha, double beta, const double *x, const double *xm1, double *xm2);
void transpose(const MATRIX *A, MATRIX *At);
void mkxm1(double *ret,const double *xm1, const double *grad, 
	   double max_svd, int n);
double tvnorm(const double *x, int imgrow, int imgcol);
void tvdenoise(const double *x, int n, double tau, int imgrow, int imgcol, double *y);
void mkdiffr1(double *zd, const double *z, int imgrow, int imgcol);
void mkdifft1(double *zd, const double *z, int imgrow, int imgcol);
void mkdiffr2(double *zd, const double *z, int imgrow, int imgcol);
void mkdifft2(double *zd, const double *z, int imgrow, int imgcol);
void mproduct(const MATRIX *A, const MATRIX *B, MATRIX *C);
double maxeigen(const MATRIX *A);
void chkdata(const int *dchk, const double *v, const double *y, const double *phi, int n);
void print_tutorial(void);

// Compile: gcc -o dttvm dttvm.c -lm
// Usage: dttvm data_file nbin resol lambda (Vgamma thresh)

int main(int argc, char **argv) {
 
	int i,j,n,m,cont_outer,iter,ist_iters,twist_iters,imgrow,imgcol,*dchk;
	double tolA, *x, f, prev_f, tau, *xm1, *xm2, max_svd, *grad, 
		alpha, beta, *y, criterion, *objective,*resid,lam1,lamN,rho0,
	        c,*phorb,r,oldvmax,*ya,*xm,vg;
	double sigma, vmax,dv,*vy,*vx,*v0,*phi,vr,old_criterion = 10.0;
	char buf[1000];
	char fname[256],dopmap[256],trail[256];
	MATRIX *A, *At;
	FILE *fp;
	char *p;

	if (argc==7) {
	        vg = atof(argv[5]);
	        tolA = atof(argv[6]);
	}
	else if (argc==6) {
 		vg = atof(argv[5]);
                tolA = 0.0001;
	}
	else if (argc==5) {
	        vg = 0.0;
		tolA = 0.0001;
	}
	else if (argc==2 && strncmp(argv[1],"-h",2)==0){
		print_tutorial();
		exit(-1);
	}
	else {
		printf("Usage: dttvm data_file nbin resol lambda (Vgamma thresh)\n");
		printf("For help: dttvm -h\n");
		exit(-1);
	}

	tau = atof(argv[4]);
  
	// Input data
	fp = fopen(argv[1],"r");
	for (i=0; fgets(buf,1000,fp)!=NULL; ++i) {}
	m = i;
	rewind(fp);
	v0 = malloc(sizeof(double)*m);
	y = malloc(sizeof(double)*m);
	phi = malloc(sizeof(double)*m);
	dchk = malloc(sizeof(int)*m);
	oldvmax = -1.0;
	for (i=0; i<m; ++i) {
		fgets(buf,1000,fp);
		dchk[i] = sscanf(buf,"%lf %lf %lf",&v0[i],&y[i],&phi[i]);
		v0[i] = v0[i] - vg;
		if (v0[i] > oldvmax)
			oldvmax = v0[i];
	}
	fclose(fp);
	chkdata(dchk,v0,y,phi,m);

	// Velocity field
	imgrow = atoi(argv[2]);
	imgcol = atoi(argv[2]);
	n = imgrow*imgcol;
	vmax = oldvmax/sqrt(2.0);
	dv = 2*vmax/(imgrow-1);
	vy = malloc(sizeof(double)*n);
	vx = malloc(sizeof(double)*n);
	for (i=0; i<imgrow; ++i){
		for (j=0; j<imgcol; ++j){
			vy[i*imgcol+j] = -vmax + dv*i;
			vx[i*imgcol+j] = -vmax + dv*j;
		}
	}

	// Make matrix
	sigma = SQ(atof(argv[3]));
	A = malloc(sizeof(MATRIX));
	A->nrow = m;
	A->ncol = n;
	A->a = malloc(sizeof(double)*m*n);
	ya = malloc(sizeof(double)*A->nrow);
	for (i=0; i<m; ++i) {
		phi[i] = 2*M_PI*phi[i];
		for (j=0; j<n; ++j) {
			vr = -vx[j]*cos(phi[i]) + vy[j]*sin(phi[i]);
			A->a[i*n+j] = 1/sqrt(2*M_PI*sigma) * exp(-SQ(v0[i]-vr)/(2*sigma));
			A->a[i*n+j] = A->a[i*n+j];
		}
	}

	// Initial image
	x = malloc(sizeof(double)*n);
	for (i=0; i<n; ++i) x[i] = 0.0;
	xm1 = malloc(sizeof(double)*n);
	xm2 = malloc(sizeof(double)*n);
	xm = malloc(sizeof(double)*n);

	// Preparing for TwIST    
	objective = malloc(sizeof(double)*MAXITER);
	resid = malloc(sizeof(double)*m);

	At = malloc(sizeof(MATRIX));
	At->nrow = n;
	At->ncol = m;
	At->a = malloc(sizeof(double)*m*n);
	grad = malloc(sizeof(double)*At->nrow);
	transpose(A,At);

	tau = tau*m/n;
	matvec(ya,A,x);
	subtract(y,ya,m,resid);
	prev_f = 0.5*innerp(resid,resid,m) + tau*tvnorm(x,imgrow,imgcol);

	cont_outer = 1;
	iter = 1;
	ist_iters = 0;
	twist_iters = 0;
	cpvector(xm2,x,n);
	cpvector(xm1,x,n);

	max_svd = 1.0;
  
	lam1 = 0.04;
	lamN = 1.0;
	rho0 = (1-lam1/lamN)/(1+lam1/lamN);
	alpha = 2/(1+sqrt(1-SQ(rho0)));
	beta = alpha*2/(lam1+lamN);

	// TwIST
	while (cont_outer){
		matvec(grad,At,resid);
		while (1) {
			mkxm1(xm,xm1,grad,max_svd,n);
			tvdenoise(xm,n,2/(tau/max_svd),imgrow,imgcol,x);
			if (ist_iters >= 2 || twist_iters != 0) {
				mkxm2(n,alpha, beta, x, xm1, xm2);
				matvec(ya,A,xm2);
				subtract(y,ya,m,resid);
				f = 0.5*innerp(resid,resid,m) + tau*tvnorm(xm2,imgrow,imgcol);
				if (f > prev_f)
					twist_iters = 0;
				else {
					++twist_iters;
					ist_iters = 0;
					cpvector(x,xm2,n);
					if ( (twist_iters % 10000) == 0){
						max_svd = 0.9*max_svd;
						fprintf(stderr,"Max_svd revised : %lf\n",max_svd);
					}
					break;
				}
			}
			else {
				matvec(ya,A,x);
				subtract(y,ya,m,resid);
				f = 0.5*innerp(resid,resid,m) + tau*tvnorm(x,imgrow,imgcol);
				if (f > prev_f){
					max_svd = 2*max_svd;
					fprintf(stderr,"Max_svd revised : %lf\n",max_svd);
					ist_iters = 0;
					twist_iters = 0;
				}
				else{
					twist_iters = twist_iters+1;
					break;
				}
			}
		}
		cpvector(xm2,xm1,n);
		cpvector(xm1,x,n);
		criterion = fabs(f-prev_f)/prev_f;
		if (iter <= MAXITER && (criterion > tolA || old_criterion > tolA)) {
			cont_outer = 1;
		}
		else {
			cont_outer = 0;
		}
		if (iter <= MINITER) {
			cont_outer = 1;
		}
		++iter;
		prev_f = f;
		objective[iter] = f;
		fprintf(stderr,"Iteration=%4d, objective=%9.5e, criterion=%7.3e\n",iter,f,criterion/tolA);
		old_criterion = criterion;
	}

	// Output
	strcpy(fname,argv[1]);
	p = strchr(fname,'.');
	if (p != NULL) {
		*p = 0;
	}
	sprintf(dopmap,"dopmap_%s.dat",fname);
	sprintf(trail,"trail_%s.dat",fname);

	fp = fopen(dopmap,"w");
	fprintf(fp,"# Dim = %d, VelRes = %8.3lf, Lambda = %e, Vgamma = %5.2lf, tolA = %e\n",imgrow,sqrt(sigma),tau*n/m, vg, tolA);
	for (i=0; i<imgrow; ++i) {
		for (j=0; j<imgcol; ++j) {
			fprintf(fp,"%lf %lf %e\n",vx[i*imgcol+j],vy[i*imgcol+j],x[i*imgcol+j]);
		}
	}
	fclose(fp);

	fp = fopen(trail,"w");
  
	// negative -> zero
	//for (i=0; i<n; ++i)
	//  if (x[i]<0) x[i]=0.0;

	matvec(ya,A,x);
	fprintf(fp,"# Dim = %d, VelRes = %8.3lf, Lambda = %e, Vgamma = %5.2lf, tolA = %e\n",imgrow,sqrt(sigma),tau*n/m, vg, tolA);
	for (i=0; i<m; ++i){
		fprintf(fp,"%lf %lf %lf %lf\n",v0[i],y[i],ya[i],phi[i]/(2*M_PI));
		if (i!=0 && i!=(m-1) && v0[i]>v0[i+1]) {
			fprintf(fp,"\n");
		}
	}
	fclose(fp);

	free(v0);
	free(y);
	free(phi);
	free(dchk);
	free(vx);
	free(vy);
	freematrix(A);
	free(x);
	free(xm1);
	free(xm2);
	free(xm);
	free(objective);
	free(resid);
	freematrix(At);
	free(ya);
	free(grad);
  
	return 1;
}

void freematrix(MATRIX *mat) {
	free(mat->a);
	free(mat);
}

void subtract(const double *x, const double *y, int n, double *sub) {
	int i;
	for (i=0; i<n; ++i) sub[i] = x[i] - y[i]; 
}

void product(double *y,const double *x, double c, int n) {
	int i;
	for (i=0; i<n; ++i) y[i] = c*x[i];
}

void matvec(double *y,const MATRIX *mat, const double *x) {
	int i,j,nr,nc;
	nr = mat->nrow;
	nc = mat->ncol;
	for (i=0; i<nr; ++i){
		y[i] = 0.0;
		for (j=0; j<nc; ++j){
			y[i] += mat->a[i*nc+j] * x[j];
		}
	}
}

double innerp(const double *x, const double *y, int n){
	int i;
	double sum=0.0;
	for (i=0; i<n; ++i) sum += x[i]*y[i];
	return sum;
}

void cpvector(double *y, const double *x, int n){
	int i;
	for (i=0; i<n; ++i) y[i] = x[i];
}

void mkxm2(int n, double alpha, double beta, 
	   const double *x, const double *xm1, double *xm2){
	int i;
	for (i=0; i<n; ++i) {
		xm2[i] = (alpha-beta)*xm1[i] + (1-alpha)*xm2[i] + beta*x[i];
	}
}

void transpose(const MATRIX *A, MATRIX *At) {
	int i,j,nrow,ncol;
	nrow = A->nrow;
	ncol = A->ncol;
	for (i=0; i<nrow; ++i) {
		for (j=0; j<ncol; ++j) {
			At->a[j*nrow+i] = A->a[i*ncol+j];
		}
	}
}

void mkxm1(double *ret,const double *xm1, const double *grad, 
	   double max_svd, int n) {
	int i;
	for (i=0; i<n; ++i) {
		ret[i] = xm1[i]+grad[i]/max_svd;
	}
}

double tvnorm(const double *x, int imgrow, int imgcol) {
	int i,j;
	double sum=0.0,*x1,*x2;
	x1 = malloc(sizeof(double)*imgrow*imgcol);
	x2 = malloc(sizeof(double)*imgrow*imgcol);
	mkdiffr1(x1,x,imgrow,imgcol);
	mkdifft1(x2,x,imgrow,imgcol);
	for (i=0; i<imgrow*imgcol; ++i)
		sum += sqrt(SQ(x1[i]) + SQ(x2[i]));
	//sum += fabs(x1[i]) + fabs(x2[i]);
	free(x1); free(x2);
	return sum;
}

void tvdenoise(const double *x, int n, double tau, int imgrow,
	       int imgcol, double *y) {
  
	int i,j,l,m, iters = 5;
	double *divp,*z,*z1,*z2,*denom,*p1,*p2,*dp1,*dp2,dt;
	double *pro;

	dt = 0.25;

	divp = malloc(sizeof(double)*n);
	z = malloc(sizeof(double)*n);
	z1 = malloc(sizeof(double)*n);
	z2 = malloc(sizeof(double)*n);
	denom = malloc(sizeof(double)*n);
	p1 = malloc(sizeof(double)*n);
	p2 = malloc(sizeof(double)*n);
	dp1 = malloc(sizeof(double)*n);
	dp2 = malloc(sizeof(double)*n);
	pro = malloc(sizeof(double)*n);

	for (i=0; i<n; ++i){
		divp[i] = 0.0;
		p1[i] = 0.0;
		p2[i] = 0.0;
	}

	for (i=0; i<iters; ++i) {
		product(pro,x,tau,n);
		subtract(divp,pro,n,z);
		mkdiffr1(z1,z,imgrow,imgcol);
		mkdifft1(z2,z,imgrow,imgcol);
		for (j=0; j<n; ++j){
			denom[j] = 1.0 + dt*sqrt(SQ(z1[j]) + SQ(z2[j]));
			//denom[j] = 1.0 + dt*(fabs(z1[j]) + fabs(z2[j]));
			p1[j] = (p1[j] + dt*z1[j])/denom[j];
			p2[j] = (p2[j] + dt*z2[j])/denom[j];
		}
		mkdiffr2(dp1,p1,imgrow,imgcol);
		mkdifft2(dp2,p2,imgrow,imgcol);
		for (j=0; j<n; ++j){
			divp[j] = dp1[j] + dp2[j];
		}
	}
	for (i=0; i<n; ++i) {
		y[i] = x[i] - divp[i]/tau;
	}
  
	free(divp); free(z); free(z1); free(z2); free(denom); 
	free(p1); free(p2); free(dp1); free(dp2); free(pro);

}

void mkdiffr1(double *zd, const double *z, int imgrow, int imgcol) {
	int l,m,nn;
	for (l=0; l<imgrow; ++l) {
		for (m=0; m<imgcol; ++m) {
			nn = l*imgcol+m;
			if (m!=imgcol-1) {
				zd[nn] = z[nn+1]-z[nn];
			}
			else {
				zd[nn] = 0.0;
			}
		}
	}
}

void mkdifft1(double *zd, const double *z, int imgrow, int imgcol) {
	int l,m,nn;
	for (l=0; l<imgrow; ++l){
		for (m=0; m<imgcol; ++m){
			nn = l*imgcol+m;
			if (l!=imgrow-1) {
				zd[nn] = z[nn+imgcol] - z[nn];
			}
			else { 
				zd[nn] = 0.0;
			}
		}
	}
}
 
void mkdiffr2(double *zd, const double *z, int imgrow, int imgcol) {
	int l,m,nn;
	for (l=0; l<imgrow; ++l){
		for (m=0; m<imgcol; ++m){
			nn = l*imgcol+m;
			if (m!=0) {
				zd[nn] = z[nn]-z[nn-1];
			}
			else {
				zd[nn] = 0.0;
			}
		}
	}
}

void mkdifft2(double *zd, const double *z, int imgrow, int imgcol) {
	int l,m,nn;
	for (l=0; l<imgrow; ++l){
		for (m=0; m<imgcol; ++m){
			nn = l*imgcol+m;
			if (l!=0) {
				zd[nn] = z[nn] - z[nn-imgcol];
			}
			else { 
				zd[nn] = 0.0;
			}
		}
	}
}
 
void mproduct(const MATRIX *A, const MATRIX *B, MATRIX *C) {
	int i,j,k,nrow,ncol;
	double sum;
	if (A->ncol != B->nrow){
		printf("Error: mproduct: ncol of A should be equal to nrow of B\n");
		exit(-1);
	}
	nrow = A->nrow;
	ncol = B->ncol;
	for (i=0; i<nrow; ++i){
		for (j=0; j<ncol; ++j){
			sum = 0.0;
			for (k=0; k<(A->ncol); ++k)
				sum += A->a[i*(A->ncol)+k] * B->a[k*(B->ncol)+j];
			C->a[i*ncol+j] = sum;
		}
	}
}

void chkdata(const int *dchk, const double *v, const double *y, 
	     const double *phi, int n) {

	int i,nnegv=0;

	for (i=0; i<n; ++i){
		if (dchk[i] != 3) {
			printf("Input data error: no data in line %d\n",i);
			exit(-1);
		}
		if (v[i]<0) ++nnegv;
	} 
	if (nnegv==0) {
		printf("Input data error: The first column should be the radial velocity.\n");
		exit(-1);
	}

	for (i=0; i<n; ++i)
		if (fabs(phi[i])>1000.0) {
			printf("Warning: phase of data No.%d is too large\n",i); 
		}
}

void print_tutorial(void) {
	printf("Doppler Tomography using Total Variation Minimization\n");
	printf("ver. 1.0, developed by M. Uemura on 130628\n");
	printf("\n");
	printf("Tutorials:\n");
	printf("Usage: dttvm data_file nbin resol lambda (Vgamma thresh)\n");
	printf("\n");
	printf("data_file: Data of the emission line, having three columns:\n");
	printf("Radial velocity in km/s, Normalized flux, and Orbital phase.\n");
	printf("The data is supposed to contain only one emission-line part.\n");
	printf("The data having multiple kinds of emission lines cannot be used.\n");
	printf("The continuum level should be zero in the Normalized flux.\n"); 
	printf("\n");
	printf("nbin: The number of bins in a side of the doppler map.\n");
	printf("The total number of bins in the doppler map is nbin*nbin.\n");
	printf("\n");
	printf("resol: Velocity resolution in the input spectra in km/s.\n");
	printf("\n");
	printf("lambda: Weight for the TVM term. Smaller values make doppler\n");
	printf("maps more noisy. Typically, 1e1 -- 1e-5.\n");
	printf("\n");
        printf("Vgamma: (optional) Gamma velocity in km/s\n");
        printf("\n");
	printf("thresh: (optional) Threshold to terminate the iteration of TwIST.\n");
	printf("The iteration is terminated when the fraction of the current to\n");
	printf("last objectives is less than this value.\n");
	printf("The default is 1e-4. A very large value may terminate the\n");
	printf("iteration before convergence. Typically, 1e-3 -- 1e-5.\n");
}
