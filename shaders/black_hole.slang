
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

[shader("compute")]
[numthreads(8, 8)]
void main(
    uint2 tid: SV_DispatchThreadID
)
{
    uint2 wh = uint2(pc.width, pc.height); 
    float4 color = float4(background_img.SampleLevel(float2(tid) / float2(wh), 0.0).rgb, 1.0);
    output_img.Store(tid, color);

}
