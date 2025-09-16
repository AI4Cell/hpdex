
#include <iostream>
#include <chrono>
#include <string>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <thread>
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
    #include <io.h>
    #define STDOUT_FILENO _fileno(stdout)
#elif defined(__APPLE__) || defined(__linux__)
    #include <sys/ioctl.h>
    #include <unistd.h>
#endif
// 获取终端宽度
static size_t get_terminal_width() {
#ifdef _WIN32
    // Windows 平台
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        return static_cast<size_t>(csbi.srWindow.Right - csbi.srWindow.Left + 1);
    }
#elif defined(__APPLE__) || defined(__linux__)
    // macOS 和 Linux 平台
    #ifdef TIOCGWINSZ
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
            return static_cast<size_t>(w.ws_col);
        }
    #endif
#endif
    return 80; // 默认宽度
}

// 根据终端宽度计算合适的进度条宽度
static size_t calculate_bar_width(const std::string& desc, const std::string& unit) {
    size_t term_width = get_terminal_width();
    
    // 估算固定文本的长度：
    // "[" + "]" + " " + "100" + "  " + "00:00<00:00, " + "99.99 " + unit + "/s"
    size_t fixed_length = desc.length() + (desc.empty() ? 0 : 1) + // 描述 + 空格
                         2 +  // "[" + "]"
                         1 +  // 空格
                         3 +  // "100"
                         2 +  // "  "
                         12 + // "00:00<00:00, "
                         6 +  // "99.99 "
                         unit.length() + 2; // unit + "/s"
    
    if (term_width <= fixed_length + 10) {
        return 10; // 最小宽度
    }
    
    return term_width - fixed_length;
}

class ProgressBar {
public:
    ProgressBar(size_t total,
              const std::string& desc = "",
              size_t width = 0,  // 0 表示自动计算
              const std::string& unit = "it",
              bool use_unicode = true)
        : total_(std::max<size_t>(1, total)),
          desc_(desc),
          width_(width == 0 ? calculate_bar_width(desc, unit) : std::max<size_t>(1, width)),
          unit_(unit),
          use_unicode_(use_unicode),
          start_time_(std::chrono::steady_clock::now()) {}

    void update(size_t current) {
        if (current > total_) current = total_;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        if (elapsed < 1e-9) elapsed = 1e-9;

        const double progress = static_cast<double>(current) / static_cast<double>(total_);
        const double exact_blocks = progress * static_cast<double>(width_);
        size_t full_blocks = static_cast<size_t>(exact_blocks);
        double frac = std::clamp(exact_blocks - static_cast<double>(full_blocks), 0.0, 1.0);
        if (full_blocks > width_) { full_blocks = width_; frac = 0.0; }

        // === 按“单元格计数”构造，避免多字节宽度问题 ===
        std::string bar;
        bar.reserve(width_ * (use_unicode_ ? 3 : 1));
        size_t cells = 0;

        const char* full_sym = use_unicode_ ? "█" : "#";
        for (size_t i = 0; i < full_blocks && cells < width_; ++i) {
            bar += full_sym; ++cells;
        }
        if (cells < width_ && frac > 0.0) {
            bar += use_unicode_ ? partial_block(frac) : ">";
            ++cells;
        }
        while (cells < width_) { bar += " "; ++cells; }

        // 速率与剩余
        const double rate = static_cast<double>(current) / elapsed;
        std::string remain_text;
        if (current < total_ && rate > 1e-12) {
            const double remain = (static_cast<double>(total_ - current) / rate);
            remain_text = fmt_time(remain);
        } else if (current < total_) {
            remain_text = "--:--";
        } else {
            remain_text = "00:00";
        }

        const int pct = static_cast<int>(std::clamp(progress * 100.0, 0.0, 100.0));

        {
            std::lock_guard<std::mutex> lock(mu_);
            std::ostringstream oss;
            if (!desc_.empty()) oss << desc_ << " ";
            oss << "[" << bar << "] "
                // tqdm 风格：elapsed<remain, rate unit/s
                << fmt_time(elapsed) << "<" << remain_text << ", "
                << std::fixed << std::setprecision(2)
                << rate << " " << unit_ << "/s";

#ifdef _WIN32
            // Windows 平台：清除当前行并回到行首
            std::cout << "\r";
            // 输出足够的空格来清除之前的内容
            static size_t last_length = 0;
            std::string output = oss.str();
            if (output.length() < last_length) {
                std::cout << std::string(last_length, ' ') << "\r";
            }
            std::cout << output << std::flush;
            last_length = output.length();
#else
            // Unix/Linux/macOS 平台：使用 ANSI 转义序列
            std::cout << "\r\033[2K" << oss.str() << std::flush;
#endif
            if (current == total_) std::cout << std::endl;
        }
    }

    // 直接完成进度条
    void complete() {
        update(total_);
    }

private:
    // tqdm 风格时间：mm:ss / hh:mm:ss / Xd hh:mm:ss
    static std::string fmt_time(double sec) {
        if (sec < 0) sec = 0;
        long long t = static_cast<long long>(sec + 0.5); // 四舍五入到秒
        long long d = t / 86400; t %= 86400;
        long long h = t / 3600;  t %= 3600;
        long long m = t / 60;    long long s = t % 60;

        std::ostringstream os;
        os << std::setfill('0');
        if (d > 0) {
            os << d << "d " << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        } else if (h > 0) {
            os << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        } else {
            os << std::setw(2) << m << ":" << std::setw(2) << s;
        }
        return os.str();
    }

    static std::string partial_block(double frac) {
        // 8 等分映射（每个占 1 单元格）
        if (frac >= 7.0/8.0) return "▉";
        if (frac >= 6.0/8.0) return "▊";
        if (frac >= 5.0/8.0) return "▋";
        if (frac >= 4.0/8.0) return "▌";
        if (frac >= 3.0/8.0) return "▍";
        if (frac >= 2.0/8.0) return "▎";
        if (frac >= 1.0/8.0) return "▏";
        return " ";
    }

    size_t total_, width_;
    std::string desc_, unit_;
    bool use_unicode_;
    std::chrono::steady_clock::time_point start_time_;
    static std::mutex mu_;
};


