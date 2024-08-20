//
// Copyright (C) 2006-2011 Christoph Sommer <christoph.sommer@uibk.ac.at>
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

#include "veins/modules/application/traci/TraCIDemo11p.h"

#include "veins/modules/application/traci/TraCIDemo11pMessage_m.h"

#include "veins/base/utils/MacToNetwControlInfo.h"

#include "veins/base/phyLayer/PhyToMacControlInfo.h"

#include "veins/modules/phy/DeciderResult80211.h"


using namespace veins;

Define_Module(veins::TraCIDemo11p);

void TraCIDemo11p::initialize(int stage)
{
    DemoBaseApplLayer::initialize(stage);
    if (stage == 0) {
        lastDroveAt = simTime();
        currentSubscribedServiceId = -1;

        TraCIDemo11pMessage* beacon = new TraCIDemo11pMessage("beacon");
        scheduleAt(simTime() + 1, beacon->dup());
    }
}

void TraCIDemo11p::onWSA(DemoServiceAdvertisment* wsa)
{
    if (currentSubscribedServiceId == -1) {
        mac->changeServiceChannel(static_cast<Channel>(wsa->getTargetChannel()));
        currentSubscribedServiceId = wsa->getPsid();
        if (currentOfferedServiceId != wsa->getPsid()) {
            stopService();
            startService(static_cast<Channel>(wsa->getTargetChannel()), wsa->getPsid(), "Mirrored Traffic Service");
        }
    }
}

void TraCIDemo11p::onWSM(BaseFrame1609_4* frame)
{
//    TraCIDemo11pMessage* wsm = check_and_cast<TraCIDemo11pMessage*>(frame);
//
//    findHost()->getDisplayString().setTagArg("i", 1, "green");
    //check if this node is in the target list
//    if(wsm->getSenderAddress()!=myId){
//        std::cout << wsm->getSenderType() <<  std::endl;
//    }

//    if(wsm->getSenderType() == 1){
//            stopService();
//            delete(wsm);
//            return;
//        }
}

void TraCIDemo11p::handleSelfMsg(cMessage* msg)
{
        // send this message on the service channel until the counter is 3 or higher.
        // this code only runs when channel switching is enabled
//        TraCIDemo11pMessage* beaconMsg = new TraCIDemo11pMessage();
//        beaconMsg->setPositionx(curPosition.x);
//        beaconMsg->setPositiony(curPosition.y);
////        std::cout << curPosition.x << "," << curPosition.y <<  std::endl;
//        beaconMsg->setTarget(beaconMsg->getSenderAddress());
//        beaconMsg->setSenderAddress(myId);
////        std::cout << myId <<  std::endl;
//        //set sender type as 1 means the msg is from a node/car/mobility
//        beaconMsg->setSenderType(1);
//        //one hot
//        beaconMsg->setSerial(3);
//        populateWSM(beaconMsg);
//        sendDown(beaconMsg);
//        std::cout << myId <<  std::endl;
//        sendDown(wsm->dup());
//        wsm->setSerial(wsm->getSerial() + 1);
//        if (wsm->getSerial() >= 3) {
//            // stop service advertisements
//            stopService();
//            delete (wsm);
//        }
//        else {
//            scheduleAt(simTime() + 1, wsm);
//        }
        // Cancel any existing scheduled event for this message
//        if (msg->isScheduled()) {
//            cancelEvent(msg);
//        }
//        // Schedule the next message
//        scheduleAt(simTime() + 1 + 0.05*myId, msg);

//    else {
//        DemoBaseApplLayer::handleSelfMsg(msg);
//    }
}

void TraCIDemo11p::handlePositionUpdate(cObject* obj)
{
    DemoBaseApplLayer::handlePositionUpdate(obj);
            TraCIDemo11pMessage* beaconMsg = new TraCIDemo11pMessage();
            beaconMsg->setPositionx(curPosition.x);
            beaconMsg->setPositiony(curPosition.y);
    //        std::cout << curPosition.x << "," << curPosition.y <<  std::endl;
            beaconMsg->setTarget(beaconMsg->getSenderAddress());
            beaconMsg->setSenderAddress(myId);
    //        std::cout << myId <<  std::endl;
            //set sender type as 1 means the msg is from a node/car/mobility
            beaconMsg->setSenderType(1);
            //one hot
            beaconMsg->setSendTime(simTime());
            beaconMsg->setSerial(3);
            populateWSM(beaconMsg);
            sendDown(beaconMsg);
//            std::cout << myId <<  std::endl;
//            sendDown(wsm->dup());
    //        wsm->setSerial(wsm->getSerial() + 1);
    //        if (wsm->getSerial() >= 3) {
    //            // stop service advertisements
    //            stopService();
    //            delete (wsm);
    //        }
    //        else {
    //            scheduleAt(simTime() + 1, wsm);
    //        }
            // Cancel any existing scheduled event for this message
    //        if (msg->isScheduled()) {
    //            cancelEvent(msg);
    //        }
    //        // Schedule the next message

//
//    // stopped for for at least 10s?
//    if (mobility->getSpeed() < 1) {
//        if (simTime() - lastDroveAt >= 10 && sentMessage == false) {
//            findHost()->getDisplayString().setTagArg("i", 1, "red");
//            sentMessage = true;
//            TraCIDemo11pMessage* wsm = new TraCIDemo11pMessage();
//            populateWSM(wsm);
//
//            // host is standing still due to crash
//            if (dataOnSch) {
//                startService(Channel::sch2, 42, "Traffic Information Service");
//                // started service and server advertising, schedule message to self to send later
//                scheduleAt(computeAsynchronousSendingTime(1, ChannelType::service), wsm);
//            }
//            else {
//                // send right away on CCH, because channel switching is disabled
//                sendDown(wsm);
//            }
//        }
//    }
//    else {
//        lastDroveAt = simTime();
//    }
}
