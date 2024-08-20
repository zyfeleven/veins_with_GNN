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

#include "veins/modules/application/traci/RSUmaster.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"


#include "veins/base/utils/MacToNetwControlInfo.h"

#include "veins/base/phyLayer/PhyToMacControlInfo.h"

#include "veins/modules/phy/DeciderResult80211.h"

#include <vector>

#include <fstream>

#include <string>

using namespace veins;

Define_Module(veins::RSUmaster);


void RSUmaster::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        //schedule the beacon message
        TraCIDemo11pMessage* beacon = new TraCIDemo11pMessage("beacon");
        scheduleAt(simTime() + 0.05 * myId, beacon);
    }
}

void RSUmaster::onWSA(DemoServiceAdvertisment* wsa)
{
    // if this RSU receives a WSA for service 42, it will tune to the chan
    if (wsa->getPsid() == 42) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
    }
}

void RSUmaster::onWSM(BaseFrame1609_4* frame)
{

}


void RSUmaster::handleSelfMsg(cMessage* msg)
{
    if (strcmp(msg->getName(), "beacon") == 0) {
//        RSUmaster::runModel();
        TraCIDemo11pMessage* beaconMsg = new TraCIDemo11pMessage();
//        // Schedule the next beacon
        scheduleAt(simTime() + 10 + 0.05 * myId, msg);
    }
    else {
        DemoBaseApplLayer::handleSelfMsg(msg);
    }
}

void RSUmaster::writePositionToFile()
{

//    std::ofstream file("./data/data.txt", std::ios::app); // Open file in append mode
//        if (file.is_open()) {
//            for (const auto& pos : positions) {
//                file << "RSU: " << pos.RSUid << " CAR: " << pos.id << " X: " << pos.x << " Y: " << pos.y << " dBm: " << pos.dbm << " Time: " << pos.time << "\n";
//            }
//            file.close();
//            // Clear the positions vector after writing to file
//            positions.clear();
////            RSUmaster::runModel();
//        } else {
//            std::cerr << "Unable to open file for writing\n";
//        }

}

void RSUmaster::runModel()
{
    std::cout << "run the model at time: "<< simTime() << std::endl;
    std::string variable = std::to_string(run_index);
    std::string CmdPyCpp = std::string("C:\\Users\\zyfel\\AppData\\Local\\Programs\\Python\\Python312\\python.exe ") +
                           "C:\\Users\\zyfel\\src\\veins\\veins-veins-5.2\\examples\\veins\\pyscript\\model.py " +
                           variable;
    system(CmdPyCpp.c_str());
    run_index += 1;
}



