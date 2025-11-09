[[vk::binding(0, 0)]]
Sampler2D background_img;

[[vk::binding(1, 0), vk::image_format("rgba8")]]
RWTexture2D<float4> output_img;

struct PushConstants
{
    uint width;
    uint height;
};

[push_constant]
ConstantBuffer<PushConstants> pc;

struct Tetrad
{
    float4 v[4];
};

struct Geodesic
{
    float4 position;
    float4 velocity;
};

static const float PI = 3.1415926535;

__generic<T : __BuiltinFloatingPointType, let N : int>
[require(glsl_spirv, GLSL_400)]
public matrix<T,N,N> inverse(matrix<T,N,N> m)
{
    __target_switch
    {
    case glsl: __intrinsic_asm "inverse";
    case spirv: return spirv_asm {
        OpExtInst $$matrix<T,N,N> result glsl450 MatrixInverse $m
    };
    }
}

interface IMetric
{
    [Differentiable]
    float4x4 get(float4 position);

    Tetrad calculate_tetrad(float4 position);

    bool is_below_event_horizon(float r);
};

[Differentiable]
float4x4 get_metric(IMetric m, float4 position)
{
   return m.get(position);
}

struct SchwarzschildMetric : IMetric
{
    static const float rs = 1.0;

    //https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.1
    [Differentiable]
    float4x4 get(float4 position)
    {
        float r = position[1];
        float theta = position[2];
    
        float4x4 m = float4x4(0.0);
        m[0][0] = -(1.0 - rs / r);
        m[1][1] = 1.0 / (1.0 - rs / r);
        m[2][2] = r * r;
        m[3][3] = r * r * sin(theta) * sin(theta);
        return m;
    }

    //https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf 2.2.6
    Tetrad calculate_tetrad(float4 position)
    {
        float r = position[1];
        float theta = position[2];
    
        float4 et = float4(rsqrt(1.0 - rs / r), 0.0, 0.0, 0.0);
        float4 er = float4(0.0, sqrt(1.0 - rs / r), 0.0, 0.0);
        float4 etheta = float4(0.0, 0.0, 1.0 / r, 0.0);
        float4 ephi = float4(0.0, 0.0, 0.0, 1.0 / (r * sin(theta)));
    
        return {et, er, etheta, ephi};
    }

    bool is_below_event_horizon(float r)
    {
        return r <= (rs + 0.0001);
    }
};

struct KerrMetric : IMetric
{
    static const float rs = 1.0;
    static const float angular_momentum = -0.5;

    [Differentiable]
    float4x4 get(float4 position)
    {
        float r = position[1];
        float theta = position[2];
        float a = angular_momentum;
        float Sigma = r*r + a*a*cos(theta)*cos(theta);
        float Delta = r*r - rs*r + a*a;

        float st2 = sin(theta) * sin(theta);

        float4x4 m = float4x4(0.0);
        m[0][0] = - (1.0 - (rs*r)/Sigma);
        m[0][3] = - 0.5 * (2.0*rs*a*r*st2) / Sigma;
        m[3][0] = - 0.5 * (2.0*rs*a*r*st2) / Sigma;
        m[1][1] = Sigma / Delta;
        m[2][2] = Sigma;
        m[3][3] = (r*r + a*a + (rs*a*a*r*st2) / Sigma) * st2;
        return m;
    }

    Tetrad calculate_tetrad(float4 position)
    {
       float r = position[1];
       float theta = position[2];
       float a = angular_momentum;
       float Sigma = r*r + a*a*cos(theta)*cos(theta);
       float Delta = r*r - rs*r + a*a;

       float4 et = float4(rsqrt(1.0 - rs*r/Sigma), 0.0, 0.0, 0.0);
       float4 er = float4(0.0, sqrt(Delta/Sigma), 0.0, 0.0);
       float4 etheta = float4(0.0, 0.0, 1.0 / sqrt(Sigma), 0.0);
       float4 ephi = float4(
           rs*a*r*sin(theta) / (sqrt(1.0 - rs*r/Sigma) * sqrt(Delta) * Sigma),
           0.0,
           0.0,
           sqrt(1.0-rs*r/Sigma) / (sqrt(Delta) * sin(theta))
       );
       return {et, er, etheta, ephi};
    }

    bool is_below_event_horizon(float r)
    {
        float r_plus = rs / 2.0 + sqrt((rs * rs) / 4 - angular_momentum * angular_momentum);
        return r <= (r_plus + 0.0001);
    }
};

float3 get_ray_through_pixel(uint sx, uint sy, uint width, uint height, float fov_degrees)
{
    float fov_rad = (fov_degrees / 360.) * 2.0 * PI;
    float f_stop = (width / 2.0) / tan(fov_rad / 2.0);

    float3 pixel_direction = float3(
        float(sx - width / 2.0), 
        float(sy - height / 2.0),
        f_stop
    );

    return normalize(pixel_direction);
}

Geodesic make_lightlike_geodesic(float4 position, float3 direction, Tetrad tetrads)
{
    Geodesic g;
    g.position = position;
    g.velocity = tetrads.v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads.v[1] * direction[0]
               + tetrads.v[2] * direction[1]
               + tetrads.v[3] * direction[2];

    return g;
}

// https://www2.mpia-hd.mpg.de/homes/tmueller/pdfs/catalogue_2014-05-21.pdf

void calculate_christoffel(IMetric m, float4 position, out float4x4 Gamma[4])
{
    float4x4 metric = get_metric(m, position);
    float4x4 metric_inverse = inverse(metric);
    float4x4 metric_diff[4]; // uses the index signature, diGjk

    for (int i = 0; i < 4; i++) {
        float4 d = { 0.0, 0.0, 0.0, 0.0 };
        d[i] = 1.0;
        DifferentialPair<float4> dp = diffPair(position, d);
        float4x4 differentiated = fwd_diff(get_metric)(m, dp).getDifferential();
        metric_diff[i] = differentiated;
    }

    [[unroll]]
    for (int mi = 0; mi < 4; mi++) {
        [[unroll]]
        for (int alpha = 0; alpha < 4; alpha++) {
            [[unroll]]
            for (int beta = 0; beta < 4; beta++) {
                float sum = 0;
                [[unroll]]
                for (int sigma = 0; sigma < 4; sigma++) {
                    sum += 0.5 * metric_inverse[mi][sigma] * (
                        metric_diff[beta][sigma][alpha] + 
                        metric_diff[alpha][sigma][beta] -
                        metric_diff[sigma][alpha][beta]
                    );
                }

                Gamma[mi][alpha][beta] = sum;
            }
        }
    }
}

float4 calculate_schwarzschild_acceleration(IMetric m, float4 X, float4 v)
{
    float4x4 gamma[4];
    calculate_christoffel(m, X, gamma);

    float4 acceleration;
    for (int mi = 0; mi < 4; mi++) {
        float4x4 gm = gamma[mi];
        acceleration[mi] = -dot(mul(gm, v), v);
        //float sum = 0.0;
        //for (int alpha = 0; alpha < 4; alpha++) {
        //    for (int beta = 0; beta < 4; beta++) {
        //        sum += -christoff2[mi][alpha][beta] * v[alpha] * v[beta];
        //    }
        //}
        //acceleration[mi] = sum;
    }
    return acceleration;
}

enum HitType
{
    ESCAPED,
    EVENT_HORIZON,
    UNFINISHED
};

struct IntegrationResult
{
    HitType type;
};

// this integrates a geodesic, until it either escapes our small universe or hits the event horizon
IntegrationResult integrate(IMetric m, inout Geodesic g)
{
    IntegrationResult result;
    result.type = HitType::UNFINISHED;

    float dt = 0.005;
    float start_time = g.position[0];

    for (int i = 0; i < 100000; i++) {
        float4 acceleration = calculate_schwarzschild_acceleration(m, g.position, g.velocity);

        g.velocity += acceleration * dt;
        g.position += g.velocity * dt;

        float radius = g.position[1];

        if (radius > 10) {
            //ray escaped
            result.type = HitType::ESCAPED;
            return result;
        }

        if (m.is_below_event_horizon(radius) || g.position[0] > (start_time + 1000)) {
            //ray has very likely hit the event horizon
            result.type = HitType::EVENT_HORIZON;
            return result;
        }
    }
    return result;
}

float2 angle_to_tex(float2 angle)
{
    float thetaf = fmod(angle[0], 2.0 * PI);
    float phif = angle[1];

    if(thetaf >= PI) {
        phif += PI;
        thetaf -= PI;
    }

    phif = fmod(phif, 2.0 * PI);

    float sxf = phif / (2.0 * PI);
    float syf = thetaf / PI;

    sxf += 0.5;
    return { sxf, syf };
}

[shader("compute")]
[numthreads(8, 8)]
void main(
    uint2 tid: SV_DispatchThreadID
)
{
    uint width = pc.width;
    uint height = pc.height;
    
    float3 ray_direction = get_ray_through_pixel(tid.x, tid.y, width, height, 90);
    float4 camera_position = { 0.0 , 8.0, PI / 2.0, -PI / 2.0};

    SchwarzschildMetric metric;
    Tetrad tetrads = metric.calculate_tetrad(camera_position);

    // the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi
    // we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    ray_direction = float3(-ray_direction[2], ray_direction[1], ray_direction[0]);

    Geodesic my_geodesic = make_lightlike_geodesic(camera_position, ray_direction, tetrads);

    IntegrationResult result = integrate(metric, my_geodesic);

    float3 color;
    if(result.type == HitType::EVENT_HORIZON || result.type == HitType::UNFINISHED) {
        color = float3(0,0,0);
    } else {
        float theta = my_geodesic.position[2];
        float phi = my_geodesic.position[3];

        float2 texcoord = angle_to_tex(float2(theta, phi));
        color = background_img.SampleLevel(texcoord, 0.0).rgb;
    }
    output_img.Store(tid, float4(color, 1.0));
}
