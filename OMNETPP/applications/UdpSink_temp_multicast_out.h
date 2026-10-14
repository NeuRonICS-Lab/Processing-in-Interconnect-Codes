//
// Copyright (C) 2011 OpenSim Ltd.
//
// SPDX-License-Identifier: LGPL-3.0-or-later
//


#ifndef __INET_UDPECHOAPP_H
#define __INET_UDPECHOAPP_H

#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"

namespace inet {

/**
 * UDP application. See NED for more info.
 */
class INET_API UdpSink_temp_multicast_out : public ApplicationBase, public UdpSocket::ICallback
{
  protected:
    enum SelfMsgKinds { TPLUS = 0, TMINUS = 1 };
    typedef std::map<int, int> SourceSequence;
    UdpSocket socket;
    inet::MacAddress macAddressCopy;
    int numEchoed; // just for WATCH
    int tout = 0;
    int tout1 = 0;
    int tout2 = 0;
    int tneg;
    int pktlen;
    simtime_t tminus; // just for WATCH
    simtime_t tplus; // just for WATCH
    simtime_t Tminus; // just for WATCH
    simtime_t Tplus; // just for WATCH
    cMessage *timerNext = nullptr;
    simtime_t startTime;
    std::string pktname;
//    int  destPort = -1;
    int tstop = 0;
//    int result1 = 0;
//    L3Address result;
    std::vector<L3Address> destAddresses;
    std::vector<std::string> destPorts;
  protected:
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;
    virtual void refreshDisplay() const override;
    virtual void processSendhp();
    virtual void processSendhm();
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;
};

} // namespace inet

#endif
