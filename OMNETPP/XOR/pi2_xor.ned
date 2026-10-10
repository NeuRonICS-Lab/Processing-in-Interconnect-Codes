//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

package inet.showcases.tsn.PI2;

//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

import inet.networks.base.TsnNetworkBase;
import inet.node.contract.IEthernetNetworkNode;
import inet.node.ethernet.EthernetLink;

network TEMP_Case_2x10x2_xor_final extends TsnNetworkBase
{
    parameters:
        //        @display("bgi=background/car;bgb=1280,720");
        *.eth[*].bitrate = default(100Mbps);
        int innodes = 2;
        int hidnodes = 10;
        int outnodes = 2;
        volatile int posX = intuniform (0,500);
    	volatile int posY = intuniform (0,500);
    submodules:
        inn[innodes]: <default("TsnDevice")> like IEthernetNetworkNode {  
            parameters: @display("p=200,80,column,100");
        }       
        ATS1: <default("TsnSwitch")> like IEthernetNetworkNode {
             parameters: @display("p=300,100");  
        }
        CBS1: <default("TsnSwitch")> like IEthernetNetworkNode {
             parameters: @display("p=400,$posX");  
        }
        hidd[hidnodes]: <default("TsnDevice")> like IEthernetNetworkNode {  
             parameters: @display("p=500,50,column,100");
        }
        
        ATS2: <default("TsnSwitch")> like IEthernetNetworkNode {
             parameters: @display("p=600,$posX");  
        }
        
        CBS2: <default("TsnSwitch")> like IEthernetNetworkNode {
             parameters: @display("p=700,100");  
        }
        
        outt[outnodes]: <default("TsnDevice")> like IEthernetNetworkNode {  
             parameters: @display("p=800,80,column,100");
        }
        tx : <default("TsnSwitch")> like IEthernetNetworkNode {  
             parameters: @display("p=900,80,column,100");
        }
        actuator : <default("TsnDevice")> like IEthernetNetworkNode {  
             parameters: @display("p=1000,80,column,100");
        }
    connections:
        for i=0..innodes-1 {
            inn[i].ethg++ <--> EthernetLink <--> ATS1.ethg++;
        }
        
        ATS1.ethg++ <--> EthernetLink <--> CBS1.ethg++;
        
        for i=0..hidnodes-1 {
            hidd[i].ethg++ <--> EthernetLink <--> CBS1.ethg++;
        }
        
        for i=0..hidnodes-1 {
            hidd[i].ethg++ <--> EthernetLink <--> ATS2.ethg++;
        }
        
        ATS2.ethg++ <--> EthernetLink <--> CBS2.ethg++;
        
        for i=0..outnodes-1 {
            outt[i].ethg++ <--> EthernetLink <--> CBS2.ethg++;
            outt[i].ethg++ <--> EthernetLink <--> tx.ethg++;
        }
        tx.ethg++ <--> EthernetLink <--> actuator.ethg++;
        
}

