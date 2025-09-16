#include <iostream>
#include <vector>
#include <string>
#include <memory>

// 简化的测试代码，专注于重现 string nullptr 问题

class SimpleProgressBar {
public:
    using Stage = std::pair<std::string, size_t>;
    
    SimpleProgressBar(const std::vector<Stage>& stages) {
        std::cout << "Creating progress bar with " << stages.size() << " stages:" << std::endl;
        for (size_t i = 0; i < stages.size(); ++i) {
            std::cout << "  Stage " << i << ": '" << stages[i].first << "' (total: " << stages[i].second << ")" << std::endl;
        }
    }
};

int main() {
    try {
        // 模拟与实际代码相同的构造过程
        std::string u_tie_name("Calc U&tie");
        std::string p_name("Calc P");
        
        std::cout << "u_tie_name: '" << u_tie_name << "'" << std::endl;
        std::cout << "p_name: '" << p_name << "'" << std::endl;
        
        std::vector<SimpleProgressBar::Stage> stages = {
            {u_tie_name, 100},
            {p_name, 200}
        };
        
        auto progress_bar = std::make_unique<SimpleProgressBar>(stages);
        
        std::cout << "Success!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
