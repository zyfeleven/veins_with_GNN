
#pragma once

#include "veins/modules/application/ieee80211p/DemoBaseApplLayer.h"
#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"
#include <vector>

namespace veins {

class VEINS_API RSUmaster : public DemoBaseApplLayer {
public:
    void initialize(int stage) override;
protected:
    typedef veins::PositionInfo PositionInfo;
    int run_index = 0;
    void onWSM(BaseFrame1609_4* wsm) override;
    void onWSA(DemoServiceAdvertisment* wsa) override;
    void handleSelfMsg(cMessage* msg) override;
    std::vector<PositionInfo> positions;
    void writePositionToFile();
    void runModel();
};

} // namespace veins
