//
// Copyright (C) 2011 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#include "inet/applications/udpapp/UdpSink_temp_multicast.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/Simsignals.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/transportlayer/common/L4PortTag_m.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/applications/base/ApplicationPacket_m.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TagBase_m.h"
#include "inet/common/TimeTag_m.h"
#include "inet/common/lifecycle/ModuleOperations.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/FragmentationTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/linklayer/common/InterfaceTag_m.h"

namespace inet {

// Global variables for shared state
MacAddress macAddressCopy;                       // MAC address to mirror
L3Address result;                                // Last resolved IP address
std::vector<L3Address> destAddresses;            // List of resolved destination addresses
std::vector<int> destport;                       // Corresponding destination ports

Define_Module(UdpSink_temp_multicast);

void UdpSink_temp_multicast::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        // init statistics
        numEchoed = 0;
        startTime = par("startTime");
        out_nodes = par("out_nodes");
        tneg = par("tneg");
        // Initial dummy values
       tstop = 0;
       tminus = 10;
       tplus = 10;
       Tminus = 10;
       Tplus = 10;
       tout = 0;
        WATCH(numEchoed);
        timerNext = new cMessage("temp_hid");
    }
}

void UdpSink_temp_multicast::handleMessageWhenUp(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        // Timer-triggered packet send logic
        if (strcmp(msg->getName(), "TPLUS") == 0) {
            EV_INFO << "calling TPLUS" << endl;
            processSendhp();
        }
        else if (strcmp(msg->getName(), "TMINUS") == 0) {
            EV_INFO << "calling TMINUS" << endl;
            processSendhm();
        }
        delete msg;
    }
    else {
        // External message from socket
        socket.processMessage(msg);
    }
}



void UdpSink_temp_multicast::socketDataArrived(UdpSocket *socket, Packet *pk)
{
    // Extract parameters
    const char *destAddrs = par("destAddresses");
    const char *destPorts = par("destPorts");
    cStringTokenizer tokenizer(destAddrs);
    cStringTokenizer tokenizer1(destPorts);

    const char *token;
    const char *token1;

    // getting the pkt name
    const char *packetName = pk->getName();
    if (packetName) {
            EV_INFO << "Packet name: " << packetName << endl;
    }
    else{
        EV_INFO << "Packet name not found " << endl;
    }
    // Identify packet by name
   EV_INFO << "Received packet: " << packetName << endl;

   // Store arrival times for xm/xp packets
   if (strcmp(packetName, "xm1") == 0 || strcmp(packetName, "xp1") == 0) {
       tplus = simTime();
       tout++;
   } else if (strcmp(packetName, "xm2") == 0 || strcmp(packetName, "xp2") == 0) {
       tminus = simTime();
       tout++;
   }

   // Once both arrive, compute TPLUS and TMINUS triggers
       if (tout == 2) {
           tplus = std::max(0.0, tneg*(tplus - tminus).dbl());
           Tplus = startTime + tplus;
           Tminus = startTime - tplus;
           scheduleAt(Tplus, new cMessage("TPLUS"));
           scheduleAt(Tminus, new cMessage("TMINUS"));
       }

       // Parse and resolve destination addresses and ports
           while ((token = tokenizer.nextToken()) && (token1 = tokenizer1.nextToken())) {
               int port = atoi(token1);
               L3Address addr;
               L3AddressResolver().tryResolve(token, addr);
               if (addr.isUnspecified()) {
                   EV_ERROR << "Cannot resolve destination: " << token << endl;
               } else {
                   destAddresses.push_back(addr);
                   destport.push_back(port);
               }
           }
           delete pk;
       }



void UdpSink_temp_multicast::processSendhm()
{
    EV_INFO << "Sending hm packets at time " << simTime() << endl;
    EV_INFO << "out_nodes = " << out_nodes << endl;

    for (int i = 0; i < 2; ++i) {
            // Create base packet
            std::string pktName = (i % 2 == 0) ? "hm1" : "hm2";
            auto *packet = new Packet(pktName.c_str());
            auto payload = makeShared<ApplicationPacket>();
            payload->setChunkLength(B(100));
            packet->insertAtBack(payload);
            packet->addTag<MacAddressInd>()->setSrcAddress(macAddressCopy);

            // Send to both destinations
            for (int j = 0; j < out_nodes; ++j) {
                Packet *dupPacket = packet->dup();
                dupPacket->setName(pktName.c_str());
                socket.sendTo(dupPacket, destAddresses[j], destport[j]);
            }

            delete packet; // Clean up original to avoid memory leak
        }

        numEchoed += 2*out_nodes;
}


void UdpSink_temp_multicast::processSendhp()
{
    EV_INFO << "Sending hp packets at time " << simTime() << endl;

    for (int i = 0; i < 2; ++i) {
            // Create base packet
            std::string pktName = (i % 2 == 0) ? "hp1" : "hp2";
            auto *packet = new Packet(pktName.c_str());
            auto payload = makeShared<ApplicationPacket>();
            payload->setChunkLength(B(100));
            packet->insertAtBack(payload);
            packet->addTag<MacAddressInd>()->setSrcAddress(macAddressCopy);

            // Send to both destinations
            for (int j = 0; j < out_nodes; ++j) {
                Packet *dupPacket = packet->dup();
                dupPacket->setName(pktName.c_str());
                socket.sendTo(dupPacket, destAddresses[j], destport[j]);
            }

            delete packet; // Clean up original to avoid memory leak
        }

        numEchoed += 2*out_nodes;
}


void UdpSink_temp_multicast::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Ignoring UDP error report " << indication->getName() << endl;
    delete indication;
}

void UdpSink_temp_multicast::socketClosed(UdpSocket *socket)
{
    if (operationalState == State::STOPPING_OPERATION)
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
}

void UdpSink_temp_multicast::refreshDisplay() const
{
    ApplicationBase::refreshDisplay();

    char buf[40];
    sprintf(buf, "echoed: %d pks", numEchoed);
    getDisplayString().setTagArg("t", 0, buf);
}

void UdpSink_temp_multicast::finish()
{

    if (timerNext != nullptr) {
        cancelAndDelete(timerNext);
        timerNext = nullptr;
    }
    ApplicationBase::finish();
}



void UdpSink_temp_multicast::handleStartOperation(LifecycleOperation *operation)
{
    socket.setOutputGate(gate("socketOut"));
    int localPort = par("localPort");
    socket.bind(localPort);
    MulticastGroupList mgl = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this)->collectMulticastGroups();
    socket.joinLocalMulticastGroups(mgl);
    socket.setCallback(this);
}

void UdpSink_temp_multicast::handleStopOperation(LifecycleOperation *operation)
{
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void UdpSink_temp_multicast::handleCrashOperation(LifecycleOperation *operation)
{
    if (operation->getRootModule() != getContainingNode(this)) // closes socket when the application crashed only
        socket.destroy(); // TODO  in real operating systems, program crash detected by OS and OS closes sockets of crashed programs.
    socket.setCallback(nullptr);
}

} // namespace inet

