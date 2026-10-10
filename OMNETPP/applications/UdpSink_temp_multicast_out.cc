//
// Copyright (C) 2011 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#include "inet/applications/udpapp/UdpSink_temp_multicast_out.h"
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
MacAddress macAddressCopy1;                       // MAC address to mirror
std::vector<L3Address> destAddresses1;            // List of resolved destination addresses
std::vector<int> destport1;                       // Corresponding destination ports

Define_Module(UdpSink_temp_multicast_out);

void UdpSink_temp_multicast_out::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        // init statistics
        numEchoed = 0;
        startTime = par("startTime");
        tneg = par("tneg");
        pktlen = par("pktlen");
        pktname = par("pktname").stringValue();
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

void UdpSink_temp_multicast_out::handleMessageWhenUp(cMessage *msg)
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



void UdpSink_temp_multicast_out::socketDataArrived(UdpSocket *socket, Packet *pk)
{
    // Extract parameters
    const char *destAddrs = par("destAddresses");
    const char *destport1s = par("destPorts");
    cStringTokenizer tokenizer(destAddrs);
    cStringTokenizer tokenizer1(destport1s);

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
   if (strcmp(packetName, "hm1") == 0 || strcmp(packetName, "hp1") == 0) {
       tplus = simTime();
       tout++;
   } else if (strcmp(packetName, "hm2") == 0 || strcmp(packetName, "hp2") == 0) {
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
                   destAddresses1.push_back(addr);
                   destport1.push_back(port);
               }
           }
           delete pk;
       }


void UdpSink_temp_multicast_out::processSendhm()
{
    EV_INFO << "Sending hm packets at time " << simTime() << endl;

    auto *packet3 = new Packet(pktname.c_str());
    auto payload = makeShared<ApplicationPacket>();
    payload->setChunkLength(B(pktlen));
    packet3->insertAtBack(payload);
    packet3->addTag<MacAddressInd>()->setSrcAddress(macAddressCopy1);
    socket.sendTo(packet3, destAddresses1[0], destport1[0]);
    numEchoed += 1;
}


void UdpSink_temp_multicast_out::processSendhp()
{
    EV_INFO << "Sending hp packets at time " << simTime() << endl;
    // Create and tag payload
    auto *packet1 = new Packet(pktname.c_str());
    auto payload = makeShared<ApplicationPacket>();
    payload->setChunkLength(B(pktlen));
    packet1->insertAtBack(payload);
    packet1->addTag<MacAddressInd>()->setSrcAddress(macAddressCopy1);
    // Send to first two destinations
    socket.sendTo(packet1, destAddresses1[0], destport1[0]);
    numEchoed += 1;
}


void UdpSink_temp_multicast_out::socketErrorArrived(UdpSocket *socket, Indication *indication)
{
    EV_WARN << "Ignoring UDP error report " << indication->getName() << endl;
    delete indication;
}

void UdpSink_temp_multicast_out::socketClosed(UdpSocket *socket)
{
    if (operationalState == State::STOPPING_OPERATION)
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
}

void UdpSink_temp_multicast_out::refreshDisplay() const
{
    ApplicationBase::refreshDisplay();
    char buf[40];
    sprintf(buf, "echoed: %d pks", numEchoed);
    getDisplayString().setTagArg("t", 0, buf);
}

void UdpSink_temp_multicast_out::finish()
{
    ApplicationBase::finish();

    if (timerNext != nullptr) {
        cancelAndDelete(timerNext);
        timerNext = nullptr;
    }
}



void UdpSink_temp_multicast_out::handleStartOperation(LifecycleOperation *operation)
{
    socket.setOutputGate(gate("socketOut"));
    int localPort = par("localPort");
    socket.bind(localPort);
    MulticastGroupList mgl = getModuleFromPar<IInterfaceTable>(par("interfaceTableModule"), this)->collectMulticastGroups();
    socket.joinLocalMulticastGroups(mgl);
    socket.setCallback(this);
}

void UdpSink_temp_multicast_out::handleStopOperation(LifecycleOperation *operation)
{
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void UdpSink_temp_multicast_out::handleCrashOperation(LifecycleOperation *operation)
{
    if (operation->getRootModule() != getContainingNode(this)) // closes socket when the application crashed only
        socket.destroy(); // TODO  in real operating systems, program crash detected by OS and OS closes sockets of crashed programs.
    socket.setCallback(nullptr);
}

} // namespace inet

