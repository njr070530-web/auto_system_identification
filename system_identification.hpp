#pragma once

#include <cstddef>
#include <cstdint>

namespace GimbalAutoId
{

enum class State : uint8_t
{
    IDLE,
    RUNNING,
    FINISHED,
    ABORTED,
};

enum class Command : uint8_t
{
    NONE,
    START,
    ABORT,
};

enum class StartError : uint8_t
{
    BUSY,
    GIMBAL_NOT_READY,
};

struct Config
{
    float duration_s;
    float buffer_ratio;
    float fade_ratio;
    float pitch_rad;
    float f_start_hz;
    float f_end_hz;
    float amplitude_max_rad;
    float omega_max_rad_s;
    float alpha_max_rad_s2;
    float current_limit;
};

struct ChirpResult
{
    float offset;
    float frequency;
    bool active;
    bool finished;
};

const Config &GetConfig();

// Called from the UART receive callback. It accepts arbitrarily split lines.
void HandleRxData(const uint8_t *data, uint16_t size);
Command ConsumeCommand();

bool Start(float now_s);
void ReportStartError(StartError error);
void Abort();
void Finish();

State GetState();
bool IsRunning();
float ElapsedTime(float now_s);
ChirpResult Evaluate(float now_s);

void RecordSample(
    uint32_t time_ms,
    float yaw_pos_ref_rad,
    float yaw_pos_fdb_rad,
    float yaw_speed_fdb_rad_s,
    int16_t yaw_current_cmd,
    bool yaw_current_saturated);

// Starts at most one DMA transfer and never waits for UART completion.
void ServiceTx();

} // namespace GimbalAutoId
