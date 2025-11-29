import glsl;

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

float4 lower_index(float4x4 metric, float4 v)
{
    return mul(metric, v);
}

float dot_metric(float4x4 metric, float4 u, float4 v)
{
    float4 lowered = lower_index(metric, u);
    return dot(lowered, v);
}

float4 gram_project(float4x4 metric, float4 u, float4 v)
{
    float top = dot_metric(metric, u, v);
    float bottom = dot_metric(metric, u, u);
    return (top / bottom) * u;
}

float4 normalize(float4x4 metric, float4 v)
{
    float d = dot_metric(metric, v, v);

    // d will be < 0 for timelike vectors
    return v * rsqrt(abs(d));
}

Tetrad gram_schmidt(float4x4 metric, float4 v0, float4 v1, float4 v2, float4 v3)
{
    float4 u0 = v0;

    float4 u1 = v1;
    u1 -= gram_project(metric, u0, u1);

    float4 u2 = v2;
    u2 -= gram_project(metric, u0, u2);
    u2 -= gram_project(metric, u1, u2);

    float4 u3 = v3;
    u3 -= gram_project(metric, u0, u3);
    u3 -= gram_project(metric, u1, u3);
    u3 -= gram_project(metric, u2, u3);

    u0 = normalize(metric, u0);
    u1 = normalize(metric, u1);
    u2 = normalize(metric, u2);
    u3 = normalize(metric, u3);

    return { u0, u1, u2, u3 };
}

float4x4 get_local_minkowski(Tetrad tetrads, float4x4 metric)
{
    // h = eT*g*e
    // matrix e makes up our tetrad vectors as column vectors

    float4x4 m;
    for (uint i = 0; i < 4; ++i) {
        m[0][i] = tetrads.v[0][i];
        m[1][i] = tetrads.v[1][i];
        m[2][i] = tetrads.v[2][i];
        m[3][i] = tetrads.v[3][i];
    }

    float4x4 minkowski;
    for (uint a = 0; a < 4; ++a) {
        for (uint b = 0; b < 4; ++b) {
            float sum = 0;
            for (uint mu = 0; mu < 4; ++mu) {
                for (uint mv = 0; mv < 4; ++mv) {
                    sum += metric[mu][mv] * m[a][mu] * m[b][mv];
                }
            }
            minkowski[a][b] = sum;
        }
    }
    return minkowski;
}

uint calculate_which_coordinate_is_timelike(Tetrad tetrads, float4x4 metric)
{
    float4x4 minkowski = get_local_minkowski(tetrads, metric);
    uint lowest_index = 0;
    float lowest_value = 0.0;

    // look for the largest negative value
    for (uint i = 0; i < 4; ++i) {
        if (minkowski[i][i] < lowest_value) {
            lowest_index = i;
            lowest_value = minkowski[i][i];
        }
    }
    return lowest_index;
}

bool approximately_equal(float v1, float v2)
{
    float bound = 0.0001;
    return v1 >= (v2 - bound) && v1 < (v2 + bound);
}

void swap<T>(inout T v1, inout T v2) where T : IArithmetic
{
    T temp = v2;
    v2 = v1;
    v1 = temp;
}

struct InitialTetradPushConstants
{
    float4 pos;
    float4 *e0;
    float4 *e1;
    float4 *e2;
    float4 *e3;
};

[shader("compute")]
[numthreads(1, 1)]
void initial_tetrad_main(uniform InitialTetradPushConstants pc)
{
    SchwarzschildMetric m;

    float4 v0 = { 1.0, 0.0, 0.0, 0.0 };
    float4 v1 = { 0.0, 1.0, 0.0, 0.0 };
    float4 v2 = { 0.0, 0.0, 1.0, 0.0 };
    float4 v3 = { 0.0, 0.0, 0.0, 1.0 };

    float4x4 metric = get_metric(m, pc.pos);

    float4 as_array[4] = { v0, v1, v2, v3 };
    float lengths[4] = { 
        dot_metric(metric, v0, v0),
        dot_metric(metric, v1, v1),
        dot_metric(metric, v2, v2),
        dot_metric(metric, v3, v3)
    };

    uint first_nonzero_idx = 0;
    for (uint i = 0; i < 4; ++i) {
        if (!approximately_equal(lengths[i], 0.0)) {
            first_nonzero_idx = i;
            break;
        }
    }

    swap(as_array[0], as_array[first_nonzero_idx]);

    float4 iv0 = as_array[0];
    float4 iv1 = as_array[1];
    float4 iv2 = as_array[2];
    float4 iv3 = as_array[3];

    Tetrad tetrad = gram_schmidt(metric, iv0, iv1, iv2, iv3);

    swap(tetrad.v[0], tetrad.v[first_nonzero_idx]);

    uint32_t timelike_coord_idx = calculate_which_coordinate_is_timelike(tetrad, metric);
    swap(tetrad.v[0], tetrad.v[timelike_coord_idx]);

    *pc.e0 = tetrad.v[0];
    *pc.e1 = tetrad.v[1];
    *pc.e2 = tetrad.v[2];
    *pc.e3 = tetrad.v[3];
#if 0
    printf("Tetrad\n"
    "%f, %f, %f, %f\n"
    "%f, %f, %f, %f\n"
    "%f, %f, %f, %f\n"
    "%f, %f, %f, %f\n",
    tetrad.v[0][0], tetrad.v[0][1], tetrad.v[0][2], tetrad.v[0][3],
    tetrad.v[1][0], tetrad.v[1][1], tetrad.v[1][2], tetrad.v[1][3],
    tetrad.v[2][0], tetrad.v[2][1], tetrad.v[2][2], tetrad.v[2][3],
    tetrad.v[3][0], tetrad.v[3][1], tetrad.v[3][2], tetrad.v[3][3]);
#endif
}

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

float4 calculate_acceleration(IMetric m, float4 X, float4 v)
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

    float dl = 0.005;
    float start_time = g.position[0];

    for (int i = 0; i < 100000; i++) {
        float4 acceleration = calculate_acceleration(m, g.position, g.velocity);

        g.velocity += acceleration * dl;
        g.position += g.velocity * dl;

        float radius = g.position[1];

        if (radius > 10) {
            //ray escaped
            result.type = HitType::ESCAPED;
            return result;
        }

        if (m.is_below_event_horizon(radius) || -g.position[0] > (start_time + 1000)) {
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

void float_controls()
{
    static const int mode = 0x1 | 0x2 | 0x4 | 0x8 | 0x10000 | 0x20000 | 0x40000;
    spirv_asm
    {
        OpExtension "SPV_KHR_float_controls2";
        OpCapability FloatControls2;
        OpExecutionModeId $main FPFastMathDefault $$float $mode;
    };
}

[shader("compute")]
[numthreads(8, 8)]
void main(
    uint2 tid: SV_DispatchThreadID
)
{
    float_controls();
    uint width = pc.width;
    uint height = pc.height;
    
    float3 ray_direction = get_ray_through_pixel(tid.x, tid.y, width, height, 90);
    float4 camera_position = { 0.0 , 7.0, PI / 2.0, -PI / 2.0};

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

static const float s_trace_dt = 0.005;

struct TraceGeodesicPushConstants
{
    float4 start_position;
    uint max_steps;
    uint pad0;
    uint pad1;
    uint pad2;
    Ptr<float4, Access.Read> start_velocity;
    float4 *out_positions;
    float4 *out_velocities;
    uint *out_steps;
};

bool geodesic_should_terminate(float4 start, float4 position, float4 velocity)
{
    bool is_broken = any(isinf(position)) || any(isinf(velocity));
    bool escaped = position[1] > 10.0;
    bool took_too_much_time = position[0] > start[0] + 1000;
    bool too_fast = abs(velocity[0]) >= 10.0;
    return escaped || took_too_much_time || too_fast || is_broken || position[1] < 0.1;
}

[shader("compute")]
[numthreads(1, 1)]
void trace_geodesic_main(uniform TraceGeodesicPushConstants pc)
{
    SchwarzschildMetric m;

    float4 position = pc.start_position;
    float4 velocity = *pc.start_velocity;

    const float dt = s_trace_dt;

    uint step = 0;
    for (step = 0; step < pc.max_steps; ++step) {
#if 0
        printf("Step %u, position {%f, %f, %f, %f}, velocity {%f, %f, %f, %f}\n",
            step,
            position.x, position.y, position.z, position.w,
            velocity.x, velocity.y, velocity.z, velocity.w);
#endif

        pc.out_positions[step] = position;
        pc.out_velocities[step] = velocity;

        float4 acceleration = calculate_acceleration(m, position, velocity);

        velocity += acceleration * dt;
        position += velocity * dt;

        if (geodesic_should_terminate(pc.start_position, position, velocity)) {
            break;
        }
    }

    *pc.out_steps = step + 1;
}

float4 parallel_transport_get_change(float4 tangent_vector, float4 geodesic_velocity, float4x4 Gamma[4])
{
    float4 dAdt;
    for (uint a = 0; a < 4; ++a) {
        float sum = 0;
        for (uint b = 0; b < 4; ++b) {
            for (uint s = 0; s < 4; ++s) {
                sum += Gamma[a][b][s] * tangent_vector[b] * geodesic_velocity[s];
            }
        }
        dAdt[a] = -sum;
    }
    return dAdt;
}

struct ParallelTransportPushConstants
{
    Ptr<float4, Access.Read> e1;
    Ptr<float4, Access.Read> e2;
    Ptr<float4, Access.Read> e3;
    Ptr<float4, Access.Read> positions;
    Ptr<float4, Access.Read> velocities;
    Ptr<uint, Access.Read> steps;
    float4 *out_e1s;
    float4 *out_e2s;
    float4 *out_e3s;
};

[shader("compute")]
[numthreads(3, 1, 1)]
void parallel_transport_tetrads(uint lid : SV_GroupThreadID, uniform ParallelTransportPushConstants pc)
{
    SchwarzschildMetric metric;
    const float dt = s_trace_dt;

    uint steps = *pc.steps;

    float4 e[3] = {
        *pc.e1,
        *pc.e2,
        *pc.e3,
    };

    Ptr<float4> out_e[3] = {
        pc.out_e1s,
        pc.out_e2s,
        pc.out_e3s,
    };

    float4 current_e = e[lid];

    for (uint i = 0; i < steps; ++i) {
        //printf("e%u s:%u {%f, %f, %f, %f}\n", lid, i, current_e.x, current_e.y, current_e.z, current_e.w);
        out_e[lid][i] = current_e;

        float4 current_position = pc.positions[i];
        float4 current_velocity = pc.velocities[i];

        float4x4 Gamma[4];
        calculate_christoffel(metric, current_position, Gamma);

        float4 e_change = parallel_transport_get_change(current_e, current_velocity, Gamma);
        current_e += e_change * dt;

    }
}
