/* Exact scalar arithmetic from R2, accelerated only by native execution.
 * GRKit preview math adaptation: MIT, copyright (c) 2026 ahonn.
 * See gr4-data/LICENSE-GRKit.txt. No photographic fitting or fast-math.
 */
#include <stddef.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    double base[256],inverse[9],matrix[9],saturation[9],banks[108],curves[768],ycc_inverse[9];
    int32_t bounds[16],selectors[20],control[21],input_encoding;
} GRConfig;

static const double ycc[9]={77./256,150./256,29./256,-43./256,-85./256,128./256,128./256,-107./256,-21./256};
static double clip(double x,double lo,double hi){return x<lo?lo:x>hi?hi:x;}
static double srgb_decode(double x){return x<=.04045?x/12.92:pow((x+.055)/1.055,2.4);}
static double srgb_encode(double x){return x<=.0031308?x*12.92:1.055*pow(fmax(x,0.),1/2.4)-.055;}
static void mul(const double*m,const double*v,double*out){
    out[0]=m[0]*v[0]+m[1]*v[1]+m[2]*v[2];
    out[1]=m[3]*v[0]+m[4]*v[1]+m[5]*v[2];
    out[2]=m[6]*v[0]+m[7]*v[1]+m[8]*v[2];
}
static double gain(double y){
    const int br[4]={0,56,768,856},start[4]={0,512,512,432},slope[4]={73,0,-7,-14};
    y=clip(y*1023,0,1023);int i=3;while(i>0&&y<br[i])--i;
    return clip(start[i]+slope[i]*(y-br[i])/8,0,1023)/512;
}
static double undo(const double*curve,double v){
    v=clip(v,curve[0],curve[255]);int lo=0,hi=254;
    while(lo<hi){int mid=(lo+hi)>>1;if(curve[mid+1]>=v)hi=mid;else lo=mid+1;}
    double a=curve[lo],b=curve[lo+1];return (lo+(b>a?clip((v-a)/(b-a),0,1):0))/(7.8/.18);
}
static double lookup(const double*curve,double x){
    x=clip(x,0,255);int i=(int)floor(x);if(i>254)i=254;double t=x-i;
    return curve[i]*(1-t)+curve[i+1]*t;
}
static void point(const double*input,double*out,const GRConfig*c){
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

__declspec(dllexport) void gr4_points(const double*src,double*dst,size_t count,const GRConfig*c){
    for(size_t p=0;p<count;++p)point(src+3*p,dst+3*p,c);
}
__declspec(dllexport) void gr4_u16(const uint16_t*src,uint16_t*dst,size_t count,const GRConfig*c,const double*exposure,double amount){
    double input[3],output[3];
    for(size_t p=0;p<count;++p){
        for(int i=0;i<3;++i)input[i]=exposure[src[3*p+i]];
        if(amount!=0)point(input,output,c);
        for(int i=0;i<3;++i){
            double base=src[3*p+i]/65535.;
            if(c->input_encoding==2)base=clip(srgb_encode(base),0,1);
            double value=amount==0?base:amount==1?output[i]:clip(base+(output[i]-base)*amount,0,1);
            dst[3*p+i]=(uint16_t)floor(value*65535+.5);
        }
    }
}
