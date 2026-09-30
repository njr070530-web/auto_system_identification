#include "system_identification.hpp"

#include "main.h"
#include "usart.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace GimbalAutoId
{
namespace
{

constexpr Config ExperimentConfig = {
    40.0f,  // duration_s
    0.2f,   // buffer_ratio
    0.1f,   // fade_ratio
    0.0f,   // pitch_rad
    1.0f,   // f_start_hz
    50.0f,  // f_end_hz
    0.05f,  // amplitude_max_rad
    1.7f,   // omega_max_rad_s
    100.0f, // alpha_max_rad_s2
    25000.0f,
};

constexpr float Pi = 3.14159265358979323846f;
constexpr size_t CommandBufferSize = 32U;
constexpr size_t SampleQueueCapacity = 256U;
constexpr size_t SamplesPerBatch = 16U;
constexpr size_t TxBufferSize = 2048U;
constexpr uint32_t FloatScale = 1000000U;
constexpr uint8_t FloatDigits = 6U;

constexpr char StartMarker[] =
    "#AUTO_ID_START,pitch_rad=0.0,T=40.0,buffer_ratio=0.2,"
    "f_start=1.0,f_end=50.0,amp_max=0.05,omega_max=1.7,"
    "alpha_max=100.0,current_limit=25000\r\n"
    "time_ms,yaw_pos_ref_rad,yaw_pos_fdb_rad,yaw_speed_fdb_rad_s,"
    "yaw_current_cmd,yaw_current_saturated\r\n";

struct Sample
{
    uint32_t time_ms;
    float yaw_pos_ref_rad;
    float yaw_pos_fdb_rad;
    float yaw_speed_fdb_rad_s;
    int16_t yaw_current_cmd;
    uint8_t yaw_current_saturated;
};

enum class TerminalMarker : uint8_t
{
    NONE,
    END,
    ABORTED,
};

enum class ErrorMarker : uint8_t
{
    NONE,
    BUSY,
    GIMBAL_NOT_READY,
};

volatile Command pending_command = Command::NONE;
char command_buffer[CommandBufferSize]{};
size_t command_length = 0U;

State state = State::IDLE;
float start_time_s = 0.0f;

Sample sample_queue[SampleQueueCapacity]{};
size_t sample_head = 0U;
size_t sample_tail = 0U;
size_t queued_samples = 0U;
uint32_t sample_count = 0U;
uint32_t dropped_sample_count = 0U;
uint32_t saturated_sample_count = 0U;

bool start_marker_pending = false;
TerminalMarker terminal_marker_pending = TerminalMarker::NONE;
ErrorMarker error_marker_pending = ErrorMarker::NONE;

alignas(32) uint8_t tx_buffers[2][TxBufferSize]
    __attribute__((section(".RAM_D1")));
uint8_t next_tx_buffer = 0U;

size_t AppendChar(char *buffer, size_t index, char value)
{
    if (index < TxBufferSize)
    {
        buffer[index] = value;
    }
    return index + 1U;
}

size_t AppendText(char *buffer, size_t index, const char *text)
{
    while (*text != '\0')
    {
        index = AppendChar(buffer, index, *text++);
    }
    return index;
}

size_t AppendUnsigned(char *buffer, size_t index, uint32_t value)
{
    char digits[10];
    size_t count = 0U;
    do
    {
        digits[count++] = static_cast<char>('0' + value % 10U);
        value /= 10U;
    } while (value != 0U && count < sizeof(digits));

    while (count > 0U)
    {
        index = AppendChar(buffer, index, digits[--count]);
    }
    return index;
}

size_t AppendSigned(char *buffer, size_t index, int32_t value)
{
    if (value < 0)
    {
        index = AppendChar(buffer, index, '-');
        value = -value;
    }
    return AppendUnsigned(buffer, index, static_cast<uint32_t>(value));
}

size_t AppendFloat(char *buffer, size_t index, float value)
{
    if (!std::isfinite(value))
    {
        return AppendText(buffer, index, "nan");
    }

    if (value < 0.0f)
    {
        index = AppendChar(buffer, index, '-');
        value = -value;
    }

    const uint32_t scaled = static_cast<uint32_t>(
        value * static_cast<float>(FloatScale) + 0.5f);
    const uint32_t integer_part = scaled / FloatScale;
    uint32_t fractional_part = scaled % FloatScale;
    index = AppendUnsigned(buffer, index, integer_part);
    index = AppendChar(buffer, index, '.');

    uint32_t divisor = FloatScale / 10U;
    for (uint8_t digit_index = 0U; digit_index < FloatDigits; ++digit_index)
    {
        const uint32_t digit = fractional_part / divisor;
        index = AppendChar(buffer, index, static_cast<char>('0' + digit));
        fractional_part %= divisor;
        divisor /= 10U;
    }
    return index;
}

size_t FormatSample(char *buffer, size_t index, const Sample &sample)
{
    index = AppendUnsigned(buffer, index, sample.time_ms);
    index = AppendChar(buffer, index, ',');
    index = AppendFloat(buffer, index, sample.yaw_pos_ref_rad);
    index = AppendChar(buffer, index, ',');
    index = AppendFloat(buffer, index, sample.yaw_pos_fdb_rad);
    index = AppendChar(buffer, index, ',');
    index = AppendFloat(buffer, index, sample.yaw_speed_fdb_rad_s);
    index = AppendChar(buffer, index, ',');
    index = AppendSigned(buffer, index, sample.yaw_current_cmd);
    index = AppendChar(buffer, index, ',');
    index = AppendChar(buffer, index, sample.yaw_current_saturated != 0U ? '1' : '0');
    index = AppendText(buffer, index, "\r\n");
    return index;
}

bool StartDmaTransfer(uint8_t *buffer, size_t length)
{
    if (length == 0U || length > TxBufferSize)
    {
        return false;
    }

    // The buffers are 32-byte aligned in AXI SRAM. Cleaning the whole fixed
    // buffer keeps the cache maintenance arguments aligned and prevents DMA
    // from reading stale formatted data.
    SCB_CleanDCache_by_Addr(
        reinterpret_cast<uint32_t *>(buffer),
        static_cast<int32_t>(TxBufferSize));
    return HAL_UART_Transmit_DMA(
               &huart7,
               buffer,
               static_cast<uint16_t>(length)) == HAL_OK;
}

void ParseCommandLine()
{
    command_buffer[command_length] = '\0';
    Command parsed = Command::NONE;
    if (std::strcmp(command_buffer, "AUTO_ID_START") == 0)
    {
        parsed = Command::START;
    }
    else if (std::strcmp(command_buffer, "AUTO_ID_ABORT") == 0)
    {
        parsed = Command::ABORT;
    }

    if (parsed != Command::NONE && pending_command == Command::NONE)
    {
        pending_command = parsed;
    }
    command_length = 0U;
}

} // namespace

const Config &GetConfig()
{
    return ExperimentConfig;
}

void HandleRxData(const uint8_t *data, uint16_t size)
{
    if (data == nullptr)
    {
        return;
    }

    for (uint16_t index = 0U; index < size; ++index)
    {
        const char value = static_cast<char>(data[index]);
        if (value == '\r' || value == '\n')
        {
            if (command_length > 0U)
            {
                ParseCommandLine();
            }
        }
        else if (command_length + 1U < CommandBufferSize)
        {
            command_buffer[command_length++] = value;
        }
        else
        {
            command_length = 0U;
        }
    }
}

Command ConsumeCommand()
{
    const uint32_t interrupt_state = __get_PRIMASK();
    __disable_irq();
    const Command command = pending_command;
    pending_command = Command::NONE;
    if (interrupt_state == 0U)
    {
        __enable_irq();
    }
    return command;
}

bool Start(float now_s)
{
    if (state == State::RUNNING)
    {
        error_marker_pending = ErrorMarker::BUSY;
        return false;
    }

    state = State::RUNNING;
    start_time_s = now_s;
    sample_head = 0U;
    sample_tail = 0U;
    queued_samples = 0U;
    sample_count = 0U;
    dropped_sample_count = 0U;
    saturated_sample_count = 0U;
    start_marker_pending = true;
    terminal_marker_pending = TerminalMarker::NONE;
    error_marker_pending = ErrorMarker::NONE;
    return true;
}

void ReportStartError(StartError error)
{
    error_marker_pending = error == StartError::BUSY
        ? ErrorMarker::BUSY
        : ErrorMarker::GIMBAL_NOT_READY;
}

void Abort()
{
    state = State::ABORTED;
    start_marker_pending = false;
    terminal_marker_pending = TerminalMarker::ABORTED;
}

void Finish()
{
    if (state != State::RUNNING)
    {
        return;
    }
    state = State::FINISHED;
    terminal_marker_pending = TerminalMarker::END;
}

State GetState()
{
    return state;
}

bool IsRunning()
{
    return state == State::RUNNING;
}

float ElapsedTime(float now_s)
{
    return now_s - start_time_s;
}

ChirpResult Evaluate(float now_s)
{
    const float elapsed_s = ElapsedTime(now_s);
    if (elapsed_s < 0.0f)
    {
        return {0.0f, 0.0f, false, false};
    }
    if (elapsed_s >= ExperimentConfig.duration_s)
    {
        return {0.0f, 0.0f, false, true};
    }

    const float buffer_time_s =
        ExperimentConfig.buffer_ratio * ExperimentConfig.duration_s;
    if (elapsed_s < buffer_time_s)
    {
        return {0.0f, 0.0f, true, false};
    }

    const float sweep_time_s = ExperimentConfig.duration_s - buffer_time_s;
    const float sweep_time = elapsed_s - buffer_time_s;
    const float progress = std::clamp(sweep_time / sweep_time_s, 0.0f, 1.0f);
    const float frequency_ratio =
        ExperimentConfig.f_end_hz / ExperimentConfig.f_start_hz;
    const float frequency = ExperimentConfig.f_start_hz *
        std::pow(frequency_ratio, progress);
    const float angular_frequency = 2.0f * Pi * frequency;
    const float amplitude_by_velocity =
        ExperimentConfig.omega_max_rad_s / angular_frequency;
    const float amplitude_by_acceleration =
        ExperimentConfig.alpha_max_rad_s2 /
        (angular_frequency * angular_frequency);
    const float amplitude = std::min(
        ExperimentConfig.amplitude_max_rad,
        std::min(amplitude_by_velocity, amplitude_by_acceleration));

    float phase;
    const float log_ratio = std::log(frequency_ratio);
    if (std::fabs(log_ratio) < 1.0e-6f)
    {
        phase = 2.0f * Pi * ExperimentConfig.f_start_hz * sweep_time;
    }
    else
    {
        phase = 2.0f * Pi * ExperimentConfig.f_start_hz * sweep_time_s /
            log_ratio * (std::pow(frequency_ratio, progress) - 1.0f);
    }

    float envelope = 1.0f;
    if (progress < ExperimentConfig.fade_ratio)
    {
        const float fade_progress = progress / ExperimentConfig.fade_ratio;
        envelope = 0.5f * (1.0f - std::cos(Pi * fade_progress));
    }
    else if (progress > 1.0f - ExperimentConfig.fade_ratio)
    {
        const float fade_progress =
            (1.0f - progress) / ExperimentConfig.fade_ratio;
        envelope = 0.5f * (1.0f - std::cos(Pi * fade_progress));
    }

    return {
        amplitude * envelope * std::sin(phase),
        frequency,
        true,
        false,
    };
}

void RecordSample(
    uint32_t time_ms,
    float yaw_pos_ref_rad,
    float yaw_pos_fdb_rad,
    float yaw_speed_fdb_rad_s,
    int16_t yaw_current_cmd,
    bool yaw_current_saturated)
{
    if (state != State::RUNNING)
    {
        return;
    }

    if (queued_samples >= SampleQueueCapacity)
    {
        ++dropped_sample_count;
        return;
    }

    sample_queue[sample_head] = {
        time_ms,
        yaw_pos_ref_rad,
        yaw_pos_fdb_rad,
        yaw_speed_fdb_rad_s,
        yaw_current_cmd,
        static_cast<uint8_t>(yaw_current_saturated ? 1U : 0U),
    };
    sample_head = (sample_head + 1U) % SampleQueueCapacity;
    ++queued_samples;
    ++sample_count;
    if (yaw_current_saturated)
    {
        ++saturated_sample_count;
    }
}

void ServiceTx()
{
    if (huart7.gState != HAL_UART_STATE_READY)
    {
        return;
    }

    uint8_t *tx_buffer = tx_buffers[next_tx_buffer];
    char *text = reinterpret_cast<char *>(tx_buffer);
    size_t length = 0U;
    size_t samples_to_remove = 0U;

    if (start_marker_pending)
    {
        length = AppendText(text, length, StartMarker);
    }
    else if (error_marker_pending != ErrorMarker::NONE)
    {
        length = AppendText(text, length, "#AUTO_ID_ERROR,");
        length = AppendText(
            text,
            length,
            error_marker_pending == ErrorMarker::BUSY
                ? "busy\r\n"
                : "gimbal_not_ready\r\n");
    }
    else if (queued_samples >= SamplesPerBatch ||
             (queued_samples > 0U && state != State::RUNNING))
    {
        samples_to_remove = std::min(queued_samples, SamplesPerBatch);
        size_t queue_index = sample_tail;
        for (size_t count = 0U; count < samples_to_remove; ++count)
        {
            length = FormatSample(text, length, sample_queue[queue_index]);
            queue_index = (queue_index + 1U) % SampleQueueCapacity;
        }
    }
    else if (terminal_marker_pending == TerminalMarker::END && queued_samples == 0U)
    {
        length = AppendText(text, length, "#AUTO_ID_STATS,samples=");
        length = AppendUnsigned(text, length, sample_count);
        length = AppendText(text, length, ",dropped=");
        length = AppendUnsigned(text, length, dropped_sample_count);
        length = AppendText(text, length, ",saturated=");
        length = AppendUnsigned(text, length, saturated_sample_count);
        length = AppendText(text, length, "\r\n#AUTO_ID_END\r\n");
    }
    else if (terminal_marker_pending == TerminalMarker::ABORTED && queued_samples == 0U)
    {
        length = AppendText(text, length, "#AUTO_ID_ABORTED\r\n");
    }

    if (length == 0U || length > TxBufferSize ||
        !StartDmaTransfer(tx_buffer, length))
    {
        return;
    }

    next_tx_buffer ^= 1U;
    if (start_marker_pending)
    {
        start_marker_pending = false;
    }
    else if (error_marker_pending != ErrorMarker::NONE)
    {
        error_marker_pending = ErrorMarker::NONE;
    }
    else if (samples_to_remove > 0U)
    {
        sample_tail = (sample_tail + samples_to_remove) % SampleQueueCapacity;
        queued_samples -= samples_to_remove;
    }
    else
    {
        terminal_marker_pending = TerminalMarker::NONE;
    }
}

} // namespace GimbalAutoId
