//
// Copyright (C) 2016 David Eckhoff <david.eckhoff@fau.de>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "veins/modules/application/traci/TraCIDemoRSU11p.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"


#include "veins/base/utils/MacToNetwControlInfo.h"

#include "veins/base/phyLayer/PhyToMacControlInfo.h"

#include "veins/modules/phy/DeciderResult80211.h"

#include <vector>

#include <fstream>

using namespace veins;

Define_Module(veins::TraCIDemoRSU11p);


void TraCIDemoRSU11p::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        //schedule the beacon message
        TraCIDemo11pMessage* beacon = new TraCIDemo11pMessage("beacon");
        scheduleAt(simTime() + 20 + 0.05 * myId, beacon->dup());
    }
}

void TraCIDemoRSU11p::onWSA(DemoServiceAdvertisment* wsa)
{
    std::cout << "test" <<  std::endl;
    // if this RSU receives a WSA for service 42, it will tune to the chan
    if (wsa->getPsid() == 42) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
    }
}

void TraCIDemoRSU11p::onWSM(BaseFrame1609_4* frame)
{
    TraCIDemo11pMessage* wsm = check_and_cast<TraCIDemo11pMessage*>(frame);
//    std::cout << "on wsm"<<  std::endl;
    if(wsm->getSenderType()==0){
//            TraCIDemo11pMessage* rsuMsg = new TraCIDemo11pMessage();
//            rsuMsg->setSenderType(2);
//            rsuMsg->setSenderAddress(myId);
//            int arraySize = positions.size();
//
//            rsuMsg->setPositionInfoArraySize(arraySize);
//            for (int i = 0; i < arraySize; ++i) {
//                veins::PositionInfo posInfo;
//                posInfo.id = positions[i].id;
//                posInfo.x = positions[i].x;
//                posInfo.y = positions[i].y;
//                posInfo.dbm = positions[i].dbm;
//                posInfo.time = positions[i].time;
//                posInfo.RSUid = myId;
//                rsuMsg->setPositionInfo(i, posInfo);
//            }
//            int arraySize = positions.size();
//            std::cout << "print , arraysize: " << arraySize <<  std::endl;

//            populateWSM(rsuMsg);
//            sendDown(rsuMsg);

//            positions.clear();
        }
    else if(wsm->getSenderType()==1){
//        std::cout << "on wsm"<<  std::endl;
//        double dbm = check_and_cast<DeciderResult80211*>(check_and_cast<PhyToMacControlInfo*>(wsm->getControlInfo())->getDeciderResult())->getRecvPower_dBm();
//        std::cout << "dbm value: " << dbm <<  std::endl;
//        std::cout << "x, y: " << wsm->getPositionx() << "," << wsm->getPositiony() <<  std::endl;
//        double posX = wsm->getPositionx();
//        double posY = wsm->getPositiony();
//        int id = wsm->getSenderAddress();
//        simtime_t time = simTime();
        PositionInfo info = {wsm->getSenderAddress(), wsm->getPositionx(), wsm->getPositiony(), check_and_cast<DeciderResult80211*>(check_and_cast<PhyToMacControlInfo*>(wsm->getControlInfo())->getDeciderResult())->getRecvPower_dBm(), simTime(), wsm->getSendTime()};
        positions.push_back(info);
    }
    //if the message is from the RSU master

    else if(wsm->getSenderType()==2){
        std::cout << "receive msg from other rsus " <<  std::endl;
    }

    // this rsu repeats the received traffic update in 2 seconds plus some random delay
//    sendDelayedDown(wsm->dup(), 2 + uniform(0.01, 0.2));
}


void TraCIDemoRSU11p::handleSelfMsg(cMessage* msg)
{
    int arraySize = positions.size();
//    std::cout << "print , arraysize: " << arraySize <<  std::endl;
    std::ofstream file("./data/data.txt", std::ios::app); // Open file in append mode
           if (file.is_open()) {
               for (const auto& pos : positions) {
                   file << "RSU: " << myId << " CAR: " << pos.id << " X: " << pos.x << " Y: " << pos.y << " dBm: " << pos.dbm << " ReceiveTime: " << pos.receive_time << " SentTime: " << pos.send_time << "\n";
               }
               file.close();
               // Clear the positions vector after writing to file
               positions.clear();
   //            RSUmaster::runModel();
           } else {
               std::cerr << "Unable to open file for writing\n";
           }
    scheduleAt(simTime() + 20 + 0.05 * myId, msg);

//    if (strcmp(msg->getName(), "beacon") == 0) {
////        TraCIDemo11pMessage* beaconMsg = new TraCIDemo11pMessage();
////        std::cout << curPosition.x << "," << curPosition.y <<  std::endl;
////        beaconMsg->setTarget(beaconMsg->getSenderAddress());
////        beaconMsg->setSenderAddress(myId);
//////        std::cout << myId <<  std::endl;
////        //set sender type as 0 means the msg is from rsu
////        beaconMsg->setSenderType(0);
////        //one hot
////        beaconMsg->setSerial(3);
////        populateWSM(beaconMsg);
////        sendDown(beaconMsg);
////        // Schedule the next beacon
//        scheduleAt(simTime() + 5, msg);
//    }
//    else {
//        std::cout << "1" <<  std::endl;
//        DemoBaseApplLayer::handleSelfMsg(msg);
//    }
}



