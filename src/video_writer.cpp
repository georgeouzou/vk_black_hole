#include "video_writer.hpp"

#include <cassert>
#include <string>
#include <format>
#include <stdexcept>
#include <windows.h>

VideoWriter::VideoWriter(uint32_t width, uint32_t height, uint32_t in_channels):
    m_width(width), m_height(height)
{
    assert(in_channels == 4);

    const char* ffmpeg_path = "ffmpeg.exe";

    int fps = 24;
    std::string cmd = std::format(
        "{} -y -f rawvideo -pixel_format rgba -video_size {}x{} -framerate {} -i - -c:v libx264 -pix_fmt yuv420p output.mp4",
        //"{} -y -f rawvideo -pixel_format rgba -video_size {}x{} -framerate {} -i - -c:v hevc_nvenc -pix_fmt yuv420p output.mp4",
        ffmpeg_path, m_width, m_height, fps);

    FILE *pipe = _popen(cmd.c_str(), "wb");
    m_pipe = pipe;
}

VideoWriter::~VideoWriter()
{
    _pclose(m_pipe);
}

void VideoWriter::write_frame(const uint8_t* data)
{
    fwrite(data, sizeof(uint8_t), m_width * m_height * 4, m_pipe);
}
