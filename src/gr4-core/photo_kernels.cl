/* Exact scalar arithmetic from R2, accelerated only by native execution.
 * GRKit preview math adaptation: MIT, copyright (c) 2026 ahonn.
 * See gr4-data/LICENSE-GRKit.txt. No photographic fitting or fast-math.
 */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

typedef struct {
    double base[256],inverse[9],matrix[9],saturation[9],banks[108],curves[768],ycc_inverse[9];
    int bounds[16],selectors[20],control[21],input_encoding;
} GRConfig;

__constant double ycc[9]={77./256,150./256,29./256,-43./256,-85./256,128./256,128./256,-107./256,-21./256};
static double clip(double x,double lo,double hi){return x<lo?lo:x>hi?hi:x;}
static double srgb_decode(double x){return x<=.04045?x/12.92:pow((x+.055)/1.055,2.4);}
static double srgb_encode(double x){return x<=.0031308?x*12.92:1.055*pow(fmax(x,0.),1/2.4)-.055;}
static void mul(__constant const double*m,const double*v,double*out){
    out[0]=m[0]*v[0]+m[1]*v[1]+m[2]*v[2];
    out[1]=m[3]*v[0]+m[4]*v[1]+m[5]*v[2];
    out[2]=m[6]*v[0]+m[7]*v[1]+m[8]*v[2];
}
static double gain(double y){
    const int br[4]={0,56,768,856},start[4]={0,512,512,432},slope[4]={73,0,-7,-14};
    y=clip(y*1023,0,1023);int i=3;while(i>0&&y<br[i])--i;
    return clip(start[i]+slope[i]*(y-br[i])/8,0,1023)/512;
}
static double undo(__constant const double*curve,double v){
    v=clip(v,curve[0],curve[255]);int lo=0,hi=254;
    while(lo<hi){int mid=(lo+hi)>>1;if(curve[mid+1]>=v)hi=mid;else lo=mid+1;}
    double a=curve[lo],b=curve[lo+1];return (lo+(b>a?clip((v-a)/(b-a),0,1):0))/(7.8/.18);
}
static double lookup(__constant const double*curve,double x){
    x=clip(x,0,255);int i=(int)floor(x);if(i>254)i=254;double t=x-i;
    return curve[i]*(1-t)+curve[i+1]*t;
}
static void point(const double*input,double*out,__constant const GRConfig*c){
    double t[3],codes[3],v[3],linear[3],colored[3],mixed[3];
    double g;
    if(c->input_encoding){
        for(int i=0;i<3;++i)linear[i]=c->input_encoding==2?input[i]:srgb_decode(input[i]);
    }else{
    mul(ycc,input,t);g=gain(t[0]);if(g<.25)g=.25;
    if(g==1){for(int i=0;i<3;++i)codes[i]=input[i];}
    else{t[1]/=g;t[2]/=g;mul(c->ycc_inverse,t,codes);}
    for(int i=0;i<3;++i)v[i]=undo(c->base,clip(codes[i],0,1)*16384);
    mul(c->inverse,v,linear);
    }
    mul(c->matrix,linear,colored);mul(ycc,colored,t);
    double y=t[0],cb=t[1],cr=t[2],a,b;int q;
    if(cr>=0){if(cb>=0){q=0;a=cr;b=cb;}else{q=3;a=-cb;b=cr;}}
    else if(cb>=0){q=1;a=cb;b=-cr;}else{q=2;a=-cr;b=-cb;}
    if(a+b>1e-12){
        double coord=q*2048+2048*b/(a+b);int quarter=(int)floor(coord/2048);if(quarter>3)quarter=3;
        const int map[4]={0,3,2,1};int quadrant=map[quarter],x=(int)floor(coord-2048*quarter),slot=0;
        while(slot<4&&x<c->bounds[quadrant*4+slot])++slot;
        int bank=c->selectors[quadrant*5+slot];
        double yy=clip(3000*y,0,4095);int br[4]={0,c->control[18],c->control[19],c->control[20]},k=3;
        while(k>0&&yy<br[k])--k;
        int slope=c->control[14+k];if(slope&0x100)slope-=0x200;
        double w=clip((c->control[10+k]+slope*(yy-br[k])/32)/1024,0,1);
        mul(c->banks+bank*9,colored,mixed);
        for(int i=0;i<3;++i)colored[i]=clip(colored[i]+w*(mixed[i]-colored[i]),0,INFINITY);
    }
    for(int i=0;i<3;++i)mixed[i]=lookup(c->curves+i*256,(7.8/.18)*colored[i])/16384;
    mul(c->saturation,mixed,v);mul(ycc,v,t);g=gain(t[0]);
    if(g==1){for(int i=0;i<3;++i)out[i]=clip(v[i],0,1);}
    else{t[1]*=g;t[2]*=g;mul(c->ycc_inverse,t,out);for(int i=0;i<3;++i)out[i]=clip(out[i],0,1);}
}


__kernel void gr4_u16(__global const ushort*src,__global ushort*dst,uint count,__constant const GRConfig*c,__global const double*exposure,double amount){
    size_t p=get_global_id(0);if(p>=count)return;
    double input[3],output[3];
    for(int i=0;i<3;++i)input[i]=exposure[src[3*p+i]];
    if(amount!=0)point(input,output,c);
    for(int i=0;i<3;++i){
        double base=src[3*p+i]/65535.;
        if(c->input_encoding==2)base=clip(srgb_encode(base),0,1);
        double v=amount==0?base:amount==1?output[i]:clip(base+(output[i]-base)*amount,0,1);
        dst[3*p+i]=(ushort)floor(v*65535+.5);
    }
}
static double lab_f(double x){double d=6./29;return x>d*d*d?cbrt(x):x/(3*d*d)+4./29;}
static double lab_inv(double f){double d=6./29;return f>d?f*f*f:3*d*d*(f-4./29);}
static void balance_pixel(double *v,double x,double y){
    double r=srgb_decode(v[0]),g=srgb_decode(v[1]),b=srgb_decode(v[2]);
    double fx=lab_f((.4124564*r+.3575761*g+.1804375*b)/.95047)-y/500;
    double fy=lab_f(.2126729*r+.7151522*g+.0721750*b);
    double fz=lab_f((.0193339*r+.1191920*g+.9503041*b)/1.08883)-x/200;
    double X=lab_inv(fx)*.95047,Y=lab_inv(fy),Z=lab_inv(fz)*1.08883;
    v[0]=clip(srgb_encode(3.2404542*X-1.5371385*Y-.4985314*Z),0,1);
    v[1]=clip(srgb_encode(-.9692660*X+1.8760108*Y+.0415560*Z),0,1);
    v[2]=clip(srgb_encode(.0556434*X-.2040259*Y+1.0572252*Z),0,1);
}
__kernel void balance_u16(__global const ushort*src,__global ushort*dst,uint count,double x,double y){
    size_t p=get_global_id(0);if(p>=count)return;
    double v[3];for(int i=0;i<3;++i)v[i]=src[3*p+i]/65535.;balance_pixel(v,x,y);
    for(int i=0;i<3;++i)dst[3*p+i]=(ushort)floor(v[i]*65535+.5);
}
__kernel void balance_u8(__global const uchar*src,__global uchar*dst,uint count,double x,double y){
    size_t p=get_global_id(0);if(p>=count)return;
    double v[3];for(int i=0;i<3;++i)v[i]=src[3*p+i]/255.;balance_pixel(v,x,y);
    for(int i=0;i<3;++i)dst[3*p+i]=(uchar)floor(v[i]*255+.5);
}
