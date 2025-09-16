#include <cstddef>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <indicators/block_progress_bar.hpp>

class ProgressTracker {
public:
    ProgressTracker(size_t nthreads)
        : nthreads_(nthreads), progress_(nthreads, 0) {
        raw_ptr_ = progress_.data();
    }

    size_t* ptr() { return raw_ptr_; }

    size_t aggregate() const {
        size_t sum = 0;
        for (size_t i = 0; i < nthreads_; ++i) {
            sum += progress_[i];
        }
        return sum;
    }

    size_t nthreads() const { return nthreads_; }

private:
    size_t nthreads_;
    std::vector<size_t> progress_;
    size_t* raw_ptr_;  // 非原子，不加锁，由外部控制
};

class ProgressBar {
public:
    struct Stage {
        const char* name_ptr;  // 裸指针（可能为 nullptr）
        size_t name_len;       // 名称长度
        size_t total;          // 总进度
    };

    ProgressBar(const std::vector<Stage>& stages,
                const ProgressTracker& tracker,
                int interval_ms = 200)
        : stages_(stages), tracker_(tracker), interval_(interval_ms), running_(false) 
    {
        if (stages_.empty()) throw std::runtime_error("No stages provided");

        // 计算 prefix sum
        prefix_totals_.resize(stages_.size());
        size_t acc = 0;
        for (size_t i = 0; i < stages_.size(); ++i) {
            acc += stages_[i].total;
            prefix_totals_[i] = acc;
        }

        current_stage_ = 0;
        init_progress_bar(stages_[0]);
    }

    void start() {
        if (running_) return;
        running_ = true;
        worker_ = std::thread([this]() {
            while (running_) {
                size_t current = tracker_.aggregate();

                // 检查是否需要切换 stage
                while (current >= prefix_totals_[current_stage_] &&
                       current_stage_ + 1 < stages_.size()) {
                    ++current_stage_;
                    init_progress_bar(stages_[current_stage_]);
                }

                // 当前 stage 的进度
                size_t stage_base = (current_stage_ == 0) ? 0 : prefix_totals_[current_stage_ - 1];
                size_t stage_progress = current - stage_base;
                size_t stage_total = stages_[current_stage_].total;

                progress_bar_->set_progress(std::min(stage_progress, stage_total));

                if (current >= prefix_totals_.back()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
            }

            // 确保完成
            progress_bar_->set_progress(stages_[current_stage_].total);
        });
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
        std::cout << std::endl;
    }

    ~ProgressBar() {
        stop();
    }

private:
    void init_progress_bar(const Stage& stage) {
        // 先结束旧的
        if (progress_bar_ && !progress_bar_->is_completed()) {
            progress_bar_->mark_as_completed();
            std::cout << std::endl;
        }

        // 安全构造前缀字符串
        std::string prefix;
        if (stage.name_ptr && stage.name_len > 0) {
            prefix.assign(stage.name_ptr, stage.name_len);
        } else {
            prefix = "Progress";
        }

        progress_bar_ = std::make_unique<indicators::BlockProgressBar>(
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::End{"]"},
            indicators::option::PrefixText{prefix + ": "},
            indicators::option::ShowPercentage{true},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::MaxProgress{stage.total},
            indicators::option::Stream{std::cout},
            indicators::option::ForegroundColor{indicators::Color::white}
        );
    }

    std::vector<Stage> stages_;
    std::vector<size_t> prefix_totals_;
    size_t current_stage_;
    const ProgressTracker& tracker_;
    int interval_;
    std::atomic<bool> running_;
    std::thread worker_;
    std::unique_ptr<indicators::BlockProgressBar> progress_bar_;
};
