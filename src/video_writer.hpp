#pragma once

#include <cstdio>
#include <cstdint>

class VideoWriter
{
public:
    VideoWriter(uint32_t width, uint32_t height, uint32_t in_channels);
    ~VideoWriter();

    void write_frame(const uint8_t* data);
private:
    FILE* m_pipe;
    uint32_t m_width;
    uint32_t m_height;
};

