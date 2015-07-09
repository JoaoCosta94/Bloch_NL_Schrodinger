void f(__global float2 *X, 
       __global float2 *OMP, 
	   __global float2 *K, 
	   int id, 
	   uint W){ 
	float2 p11, p22, p33, p21, p31, p32, op, aux;
	p11 = X[id*W];
	p22 = X[id*W+1];
	p33 = X[id*W+2];
	p21 = X[id*W+3];
	p31 = X[id*W+4];
	p32 = X[id*W+5];
	op = OMP[id];
	op = complex_mul(op, complex_unit);
	
	aux = p22 * gama/2.0 + complex_mul(op, p22) + conj(p22) * gama/2.0 + complex_mul(op, conj(p22));
	K[id*W] = aux;
	
	aux = (-p22*gama - complex_mul(op, p21) + complex_mul(p32, complex_unit)*omc 
	      - conj(p22)*gama - complex_mul(op, conj(p21)) + complex_mul(conj(p32), complex_unit)*omc);
	K[id*W+1] = aux;
	
	aux = p22*gama/2.0 - complex_mul(p32, complex_unit)*omc + conj(p22)*gama/2.0 - complex_mul(conj(p32), complex_unit)*omc;
	K[id*W+2] = aux;
	
	aux = complex_mul(op, p11) - complex_mul(op, p22) - p21*gama) + complex_mul(p21, complex_unit)*delta + complex_mul(p31, complex_unit)*omc;	 
	K[id*W+3] = aux;
	
	aux = complex_mul(p21, complex_unit)*omc + complex_mul(p31, complex_unit)*delta - complex_mul(op, p32);
	K[id*W+4] = aux;
	
	aux = (complex_mul(p22, complex_unit)*omc - omplex_mul(p33, complex_unit)*omc - complex_mul(omp, p31) - p32*gama);
	K[id*W+5] = aux;
}

__kernel void RK4Step(__global float2 *X, 
				      __global float2 *OMP, 
					  __global float2 *K, 
					  __global float2 *Xs, 
					  __global float2 *Xm, 
					  uint W){
    const int gid_x = get_global_id(0);
	int idx = 0;

    //computation of k1
    f(X, OMP, K, gid_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = X[idx] + dt*K[idx]/6.0;
		Xm[idx] = X[idx] + 0.5*dt*K[idx];
	}
    
    //computation of k2
    f(Xm, OMP, K, gid_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/3.0;
		Xm[idx] = X[idx] + 0.5*dt*K[idx];
	}	

    //computation of k3
    f(Xm, OMP, K, gid_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/3.0;
		Xm[idx] = X[idx] + dt*K[idx];
	}	

    //computation of k4
    f(Xm, OMP, K, gid_x, W);
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		Xs[idx] = Xs[idx] + dt*K[idx]/6.0;
	}

    //update photon
	for(int i=0; i<W; i++)
	{
		idx = gid_x*W+i;
		X[idx] = Xs[idx];
	}
}