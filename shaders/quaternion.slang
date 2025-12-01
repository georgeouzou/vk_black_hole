module quaternion;

float4 quat_from_axis_angle(float3 axis, float angle)
{
    const float PI = 3.1415926535;
    float4 qr;
    float half_angle = (angle * 0.5) * PI / 180.0;
    qr.x = axis.x * sin(half_angle);
    qr.y = axis.y * sin(half_angle);
    qr.z = axis.z * sin(half_angle);
    qr.w = cos(half_angle);
    return qr;
}

float4 quat_mult(float4 q1, float4 q2)
{
    float4 qr;
    qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
    return qr;
}

public float3 quat_rotate(float3 v, float3 axis, float angle)
{
    float4 q = quat_from_axis_angle(axis, angle);
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

