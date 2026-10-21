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
#define NCV (10)

typedef struct{
	int nrow;
	int ncol;
	double *a;
} MATRIX;

void freematrix(MATRIX *mat);
void subtract(const double *x, const double *y, int n, double *sub);
void product(double *y,const double *x, double c, int n);
void matvec(double *y,const MATRIX *mat, const int nrow, const int ncol, const double *x);
double innerp(const double *x, const double *y, int n);
void cpvector(double *y, const double *x, int n);
void mkxm2(int n, double alpha, double beta, const double *x, const double *xm1, double *xm2);
void transpose(const MATRIX *A, const int nrow, const int ncol, MATRIX *At);
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
void mksubdata(const int id, const MATRIX *A, const double *y, const int *subid, MATRIX *At, double *yt, int *mt, MATRIX *Av, double *yv, int *mv);
void dttvm_main(const MATRIX *A, const double *y, const int m, const int n, const double tau0, const int imgrow, const int imgcol, const double tolA, double *x);
void chkdata(const int *dchk, const double *v, const double *y, const double *phi, int n);
void print_tutorial(void);
void init_genrand(unsigned long s);
double genrand_real1(void);

/* X-ray response: fixed line ratios and a common velocity map. */
#define C_KMS 299792.458

typedef struct{
	double energy, weight, fwhm_ev, offset, a;
} LINE;

/* Smith et al. (2015), Appendix A1: approximate Voigt H(a,x). */
static inline double H_smith15(double a, double x)
{
    /* coefficients (Table A1) */
    const double A0=15.75328153963877,  A1=286.9341762324778, A2=19.05706700907019;
    const double A3=28.22644017233441,  A4=9.526399802414186, A5=35.29217026286130;
    const double A6=0.8681020834678775;
    const double B0=0.0003300469163682737, B1=0.5403095364583999, B2=2.676724102580895;
    const double B3=12.82026082606220,    B4=3.21166435627278,  B5=32.032981933420;
    const double B6=9.0328158696,         B7=23.7489999060,     B8=1.82106170570;

    double z = x*x;
    if (a == 0.0) return exp(-z);
    if (z <= 3.0)
        return exp(-z) * (1.0 - a*(A0 + A1/(z - A2 + A3/(z - A4 + A5/(z - A6)))));
    if (z < 25.0)
        return exp(-z) + a*(B0 + B1/(z - B2 + B3/(z + B4 + B5/(z - B6 + B7/(z - B8)))));
    return 1.0 / sqrt(M_PI) * a / (z - 1.5 - 1.5/(z - 3.5 - 5.0/(z - 5.5)));
}

static double voigt_profile(double dv, double sigma, double a)
{
	/* sigma is the instrumental Gaussian variance, as in the original code. */
	if (a == 0.0)
		return 1/sqrt(2*M_PI*sigma) * exp(-SQ(dv)/(2*sigma));
	return H_smith15(a, dv/sqrt(2*sigma))/sqrt(2*M_PI*sigma);
}

static int read_line(FILE *fp, LINE *line)
{
	char buf[1000], *p, extra;
	while (fgets(buf,sizeof(buf),fp) != NULL) {
		p = strchr(buf,'#');
		if (p != NULL) *p = 0;
		p = buf;
		while (isspace((unsigned char)*p)) ++p;
		if (*p == 0) continue;
		if (sscanf(p,"%lf %lf %lf %c",&line->energy,&line->weight,
			   &line->fwhm_ev,&extra) != 3 ||
			!isfinite(line->energy) || line->energy <= 0 ||
			!isfinite(line->weight) || line->weight < 0 ||
			!isfinite(line->fwhm_ev) || line->fwhm_ev < 0) {
			fprintf(stderr,"Line list: expected energy[keV]>0, weight>=0, natural FWHM[eV]>=0\n");
			exit(-1);
		}
		return 1;
	}
	return 0;
}

// Compile: gcc -o dttvm_xray_cv dttvm_xray_cv.c mt19937.c gasdev.c -lm
// Usage: dttvm_xray_cv data_file nbin resol lambda_min lambda_max nlambda lines Eref (Vgamma thresh)

int main(int argc, char **argv) {
 
	int i,j,n,m,mt,mv,imgrow,imgcol,*dchk,nsub,*subid,ncv;
	double tolA, *x,tau, tau0, tau1, dtau, *y,*yv,*yt,*yvm,c,*phorb,r,oldvmax,vg;
	double sigma, vmax,dv,*vy,*vx,*v0,*phi,vr,sum,ssum,ave,sig,*resid,cvdev;
	char buf[1000];
	MATRIX *A,*Av,*At;
	FILE *fp;
	char *p;
	int k,nline;
	double eref,wsum;
	LINE line,*lines;

	if (argc==11) {
	        vg = atof(argv[9]);
	        tolA = atof(argv[10]);
	}
	else if (argc==10) {
 		vg = atof(argv[9]);
                tolA = 0.0001;
	}
	else if (argc==9) {
	        vg = 0.0;
		tolA = 0.0001;
	}
	else if (argc==2 && strncmp(argv[1],"-h",2)==0){
		print_tutorial();
		exit(-1);
	}
	else {
		printf("Usage: dttvm_xray_cv data_file nbin resol lambda_min lambda_max nlambda lines Eref (Vgamma thresh)\n");
		printf("For help: dttvm_xray_cv -h\n");
		exit(-1);
	}

	// Line list: rest energy [keV], relative weight, natural FWHM [eV].
	eref = atof(argv[8]);
	if (!isfinite(eref) || eref <= 0 ||
		!isfinite(atof(argv[3])) || atof(argv[3]) <= 0) {
		fprintf(stderr,"Eref and instrumental Gaussian sigma must be positive.\n");
		exit(-1);
	}
	fp = fopen(argv[7],"r");
	if (fp == NULL) { perror(argv[7]); exit(-1); }
	nline = 0;
	while (read_line(fp,&line)) ++nline;
	if (nline == 0) {
		fprintf(stderr,"Empty line list.\n");
		exit(-1);
	}
	rewind(fp);
	lines = malloc(sizeof(LINE)*nline);
	if (lines == NULL) { fprintf(stderr,"Cannot allocate line list.\n"); exit(-1); }
	wsum = 0.0;
	for (k=0; k<nline; ++k) {
		if (!read_line(fp,&lines[k])) { fprintf(stderr,"Incomplete line list.\n"); exit(-1); }
		/* Positive velocity denotes redshift: v = c*(Eref/E - 1). */
		lines[k].offset = C_KMS*(eref/lines[k].energy-1.0);
		/* Local narrow-line conversion: Lorentz HWHM in km/s divided by sqrt(2)*sigma_inst. */
		lines[k].a = C_KMS*(lines[k].fwhm_ev/1000.0)/(2*lines[k].energy)
			/(sqrt(2.0)*atof(argv[3]));
		wsum += lines[k].weight;
	}
	fclose(fp);
	if (!isfinite(wsum) || wsum <= 0) {
		fprintf(stderr,"Sum of line weights must be positive and finite.\n");
		exit(-1);
	}
	for (k=0; k<nline; ++k) lines[k].weight /= wsum;

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
	for (i=0; i<m; ++i) {
		phi[i] = 2*M_PI*phi[i];
		for (j=0; j<n; ++j) {
			vr = -vx[j]*cos(phi[i]) + vy[j]*sin(phi[i]);
			A->a[i*n+j] = 0.0;
			for (k=0; k<nline; ++k) {
				if (lines[k].weight == 0.0) continue;
				A->a[i*n+j] += lines[k].weight
					* voigt_profile(v0[i]-vr-lines[k].offset,sigma,lines[k].a);
			}
		}
	}

	// Define sub-sets
	nsub = 10;
	unsigned long ss = time( NULL ); // Initialize random number.
	init_genrand(ss);
	subid = malloc(sizeof(int)*m);
	for (i=0; i<m; ++i){
		r = genrand_real1();
		for (j=0; j<nsub; ++j){
			if ( (double)j/(double)nsub <= r && r < (double)(j+1)/(double)nsub ) 
				subid[i] = j;
		}
	}

	// C.V. DTTVM
	tau0 = atof(argv[4]);
	tau1 = atof(argv[5]);
	ncv  = atof(argv[6]);
	dtau = (log10(tau1)-log10(tau0))/(ncv-1);
	At = malloc(sizeof(MATRIX));
	Av = malloc(sizeof(MATRIX));
	At->nrow = m;
	At->ncol = n;
	Av->nrow = m;
	Av->ncol = n;
	At->a = malloc(sizeof(double)*m*n);
	Av->a = malloc(sizeof(double)*m*n);
	yt = malloc(sizeof(double)*A->nrow);
	yv = malloc(sizeof(double)*A->nrow);
	yvm = malloc(sizeof(double)*A->nrow);
	x = malloc(sizeof(double)*n);
	resid = malloc(sizeof(double)*m);
	
#pragma omp parallel for 
	for (i=0; i<ncv; ++i) {
	        tau = pow(10,log10(tau0) + dtau*i);
		sum = 0.0;
		ssum = 0.0;
		for (j=0; j<nsub; ++j) {
			mksubdata(j,A,y,subid,At,yt,&mt,Av,yv,&mv);
			dttvm_main(At,yt,mt,n,tau,imgrow,imgcol,tolA,x); // TVM
			matvec(yvm,Av,mv,n,x);
			subtract(yv,yvm,mv,resid);
			cvdev = innerp(resid,resid,mv)/mv;
			sum += cvdev;
			ssum += SQ(cvdev);
			fprintf(stderr,"Lambda ID %02d/%d  Sub-Set ID %02d/%d (Nt=%d, Nv=%d) err=%e\n",i+1,ncv,j+1,nsub,mt,mv,cvdev);
		}
		ave = sum/nsub;
		sig = sqrt(ssum-nsub*ave*ave)/nsub;
		printf("%e %e %e\n",tau,ave,sig);
	}	
	free(x);
	freematrix(A);
	freematrix(At);
	freematrix(Av);
	free(v0);
	free(y);
	free(yt);
	free(yv);
	free(yvm);
	free(phi);
	free(dchk);
	free(vy);
	free(vx);
	free(resid);
	free(subid);
	free(lines);
}

void dttvm_main(const MATRIX *A, const double *y, const int m, const int n, const double tau0, const int imgrow, const int imgcol, const double tolA, double *x){

	int i,cont_outer, iter, ist_iters = 0, twist_iters;	
	double *xm1,*xm2,*xm,*objective,*resid,*grad,tau,*ya,f,prev_f,max_svd,
	  lam1,lamN,rho0,alpha,beta,criterion,old_criterion=10.0;
	MATRIX *At;

	// Initial image
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
	grad = malloc(sizeof(double)*n);
	transpose(A,m,n,At);

	tau = tau0*m/n;
	ya = malloc(sizeof(double)*m);
	matvec(ya,A,m,n,x);
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
		matvec(grad,At,n,m,resid);
		while (1) {
			mkxm1(xm,xm1,grad,max_svd,n);
			tvdenoise(xm,n,2/(tau/max_svd),imgrow,imgcol,x);
			if (ist_iters >= 2 || twist_iters != 0) {
				mkxm2(n,alpha, beta, x, xm1, xm2);
				matvec(ya,A,m,n,xm2);
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
				matvec(ya,A,m,n,x);
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
		// fprintf(stderr,"Iteration=%4d, objective=%9.5e, criterion=%7.3e\n",iter,f,criterion/tolA);
		old_criterion = criterion;
	}

	free(xm1);
	free(xm2);
	free(xm);
	free(objective);
	free(resid);
	freematrix(At);
	free(ya);
	free(grad);
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

void matvec(double *y,const MATRIX *mat, const int nrow, const int ncol,const double *x) {
	int i,j,nr,nc;
	for (i=0; i<nrow; ++i){
		y[i] = 0.0;
		for (j=0; j<ncol; ++j){
			y[i] += mat->a[i*ncol+j] * x[j];
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

void transpose(const MATRIX *A, const int nrow, const int ncol, MATRIX *At) {
	int i,j;
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

void mksubdata(const int id, const MATRIX *A, const double *y, const int *subid, MATRIX *At, double *yt, int *mt, MATRIX *Av, double *yv, int *mv) {

  int i,j,k,l,m,n;

  m = A->nrow;
  n = A->ncol;
  j = 0;
  k = 0;
  for (i=0; i<m; ++i) {
    if ( subid[i] == id ) {
      yv[j] = y[i];
      for (l=0; l<n; ++l) 
	Av->a[j*n+l] = A->a[i*n+l];
      ++j;
    }
    else {
      yt[k] = y[i];
      for (l=0; l<n; ++l) 
	At->a[k*n+l] = A->a[i*n+l];
      ++k;
    }
  }
  *mt = k;
  *mv = j;
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
	printf("Cross-validation for Doppler Tomography using Total Variation Minimization\n");
	printf("ver. 1.0, developed by M. Uemura on 141219\n");
	printf("\n");
	printf("Tutorials:\n");
	printf("Usage: dttvm_xray_cv data_file nbin resol lambda_min lambda_max nlambda lines Eref (Vgamma thresh)\n");
	printf("\n");
	printf("data_file: Data of the emission line, having three columns:\n");
	printf("Radial velocity in km/s, Normalized flux, and Orbital phase.\n");
	printf("The lines share one velocity map with fixed intensity ratios.\n");
	printf("Intrinsic thermal and turbulent motions are included in the map.\n");
	printf("The continuum level should be zero in the Normalized flux.\n"); 
	printf("\n");
	printf("nbin: The number of bins in a side of the doppler map.\n");
	printf("The total number of bins in the doppler map is nbin*nbin.\n");
	printf("\n");
	printf("resol: Instrumental Gaussian sigma in km/s (not FWHM).\n");
	printf("\n");
	printf("lambda_min: Minimum weight for the TVM term.\n");
	printf("\n");
	printf("lambda_max: Maximum weight for the TVM term.\n");
	printf("\n");
	printf("nlambda: The number of points in the cross-validation curve.\n");
	printf("\n");
	printf("lines: File with rest energy [keV], relative weight, natural FWHM [eV].\n");
	printf("Blank lines and # comments are allowed in the line list.\n");
	printf("Eref: Reference energy [keV] for input v=c*(Eref/E-1), positive=redshift.\n");
	printf("Voigt uses the Smith et al. (2015) approximation; Gaussian width is instrumental only.\n");
	printf("\n");
        printf("Vgamma: (optional) Gamma velocity in km/s\n");
        printf("\n");
	printf("thresh: (optional) Threshold to terminate the iteration of TwIST.\n");
	printf("The iteration is terminated when the fraction of the current to\n");
	printf("last objectives is less than this value.\n");
	printf("The default is 1e-4. A very large value may terminate the\n");
	printf("iteration before convergence. Typically, 1e-3 -- 1e-5.\n");
}
