#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

#include "TCanvas.h"
#include "TPad.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TLatex.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TGaxis.h"
#include "TString.h"

#include "globalVars.h"

void insertCell(std::map<std::pair<double,double>,double> & lMap,
		TH2D* & hMap,
		const unsigned iBX,
		const unsigned iBY,
		const double aLumi,
		const bool addContent = true){
  
  std::pair<std::map<std::pair<double,double>,double>::iterator,bool> lInsert = lMap.insert(std::pair<std::pair<double,double>,double>(std::pair<double,double>(hMap->GetXaxis()->GetBinCenter(iBX),hMap->GetYaxis()->GetBinCenter(iBY)),aLumi));
  
  if (addContent && !lInsert.second) {
    double & lTmp = lInsert.first->second;
    //if (lTmp.find(year[iY])==lTmp.npos) lTmp += year[iY];
    //lTmp += run[iY][iR];
    lTmp += aLumi;
  }
};

unsigned convertToInt(std::string lS){
  unsigned tmp = 0;

  size_t pos1 = lS.find("17");
  size_t pos2 = lS.find("18");
  if (pos2 == lS.npos) pos2=lS.size();
  std::string lruns1 = pos1 == lS.npos ? "" : lS.substr(pos1+2,pos2-pos1-2);
  std::string lruns2 = pos2 == lS.size() ? "" : lS.substr(pos2+2,lS.size()-pos2-2);
  
  //std::cout << lS << " npos " << lS.size() << " pos1 " << pos1 << " pos2 " << pos2 << " runs17: " << lruns1 << " runs18: " << lruns2 << std::endl;

  if (lruns1.find("B")!=lruns1.npos) tmp = tmp | (0x1);
  if (lruns1.find("C")!=lruns1.npos) tmp = tmp | (1<<1);
  if (lruns1.find("DE")!=lruns1.npos) tmp = tmp | (0x1<<2);
  if (lruns1.find("F")!=lruns1.npos) tmp = tmp | (0x1<<3);
  if (lruns2.find("A")!=lruns2.npos) tmp = tmp | (0x1<<4);
  if (lruns2.find("B")!=lruns2.npos) tmp = tmp | (0x1<<5);
  if (lruns2.find("C")!=lruns2.npos) tmp = tmp | (0x1<<6);
  if (lruns2.find("D")!=lruns2.npos) tmp = tmp | (0x1<<7);

  std::cout << lS << " hex code " << std::hex << tmp << std::endl;
  
  return tmp;
  
};

int plotHotTowers(){


  SetTdrStyle();
  gStyle->SetOptStat(0);
  
  TLatex lat;
  char buf[300];

  const bool do17 = false;
  const bool do18 = true;
  const bool doCold = true;
  
  const unsigned nDR = 2;
  float dRcut[nDR] = {-1,0.2};
  std::string DR[nDR] = {"dR0","dR2"};

  const unsigned nY = 2;
  std::string yearShort[2] = {"17","18"};

  const unsigned nR = 4;

  double lumiTot = 41557+59970;
  
  TFile *f[nY][nR];
  TH2D *hMap[nY][nR];
  TCanvas *myc1 = new TCanvas("myc1","myc1",1);
  TCanvas *myc2 = new TCanvas("myc2","myc2",1);

  for (unsigned idr(0); idr<nDR; ++idr){
    
    
    std::map<std::pair<double,double>,double> lMap;
    std::map<std::pair<double,double>,double> lMapRun;
    
    for (unsigned iY(0); iY<nY; ++iY){//loop on year
      std::cout << " -- " << year[iY] << std::endl;
      if (do17 && !do18 && iY==1) continue;
      else if (!do17 && do18 && iY==0) continue;
      for (unsigned iR(0); iR<nR; ++iR){//loop on run
	std::cout << " -- run " << runPeriod[iY][iR] << std::endl;
	
	if (!doCold) f[iY][iR] = TFile::Open(("hotjets-"+yearShort[iY]+"run"+runPeriod[iY][iR]+".root").c_str());
	else f[iY][iR] = TFile::Open(("coldjets-"+yearShort[iY]+"run"+runPeriod[iY][iR]+".root").c_str());
	if (!f[iY][iR]) return 1;
	f[iY][iR]->cd();
	hMap[iY][iR] = (TH2D*)gDirectory->Get(doCold?"h2hole":"h2hotfilter");
	if (!hMap[iY][iR]) return 1;
	
	myc1->cd();
	gPad->SetRightMargin(0.15);
	gPad->SetGridx(1);
	gPad->SetGridy(1);
	hMap[iY][iR]->Draw("colz");
	
	double stepX = (hMap[iY][iR]->GetXaxis()->GetBinLowEdge(2)-hMap[iY][iR]->GetXaxis()->GetBinLowEdge(1));
	double stepY = (hMap[iY][iR]->GetYaxis()->GetBinLowEdge(2)-hMap[iY][iR]->GetYaxis()->GetBinLowEdge(1));
	
	std::cout << " nBins = " << hMap[iY][iR]->GetNbinsX() << " " << hMap[iY][iR]->GetNbinsY() << " step sizes: " << stepX << " " << stepY << std::endl;
	
	lMapRun.clear();
	
	
	for (int iBX(1); iBX<hMap[iY][iR]->GetNbinsX()+1;++iBX){
	  for (int iBY(1); iBY<hMap[iY][iR]->GetNbinsY()+1;++iBY){
	    if (hMap[iY][iR]->GetBinContent(iBX,iBY)>0){
	      //std::cout << hMap[iY][iR]->GetXaxis()->GetBinLowEdge(iBX) << " "
	      //	<< hMap[iY][iR]->GetXaxis()->GetBinLowEdge(iBX+1) << " "
	      //	<< hMap[iY][iR]->GetYaxis()->GetBinLowEdge(iBY) << " "
	      //	<< hMap[iY][iR]->GetYaxis()->GetBinLowEdge(iBY+1)
	      //	<< std::endl;
	      //std::pair<std::map<std::pair<double,double>,double>::iterator,bool> lInsert = lMap.insert(std::pair<std::pair<double,double>,double>(std::pair<double,double>(hMap[iY][iR]->GetXaxis()->GetBinCenter(iBX),hMap[iY][iR]->GetYaxis()->GetBinCenter(iBY)),year[iY]+run[iY][iR]));
	      if (dRcut[idr]<0) {
		insertCell(lMap,hMap[iY][iR],iBX,iBY,lumiRun[iY][iR],true);
		insertCell(lMapRun,hMap[iY][iR],iBX,iBY,lumiRun[iY][iR],true);
	      }
	      else {
		//add bins around it within dR = dRcut
		for (int iNeigh(0); iNeigh<10;++iNeigh){
		  if ( (iBX-iNeigh)<0 || (iBX+iNeigh)>hMap[iY][iR]->GetNbinsX() ) continue;
		  for (int jNeigh(0); jNeigh<10;++jNeigh){
		    if ( (iBY-jNeigh)<0 || (iBY+jNeigh)>hMap[iY][iR]->GetNbinsY() ) continue;
		    double dR = sqrt(pow(stepX*iNeigh,2)+pow(stepY*jNeigh,2));
		    if (dR >= dRcut[idr]) continue;
		    //std::cout << " -- adding neigh "
		    //	      << iNeigh << "," << jNeigh
		    //	      << " dR = " << dR
		    //	      << " bin below: " << iBX-iNeigh << "," << iBY-jNeigh
		    //	      << " bin above: " << iBX+iNeigh << "," << iBY+jNeigh
		    //	      << std::endl;
		    insertCell(lMapRun,hMap[iY][iR],iBX-iNeigh,iBY-jNeigh,lumiRun[iY][iR],false);
		    insertCell(lMapRun,hMap[iY][iR],iBX-iNeigh,iBY+jNeigh,lumiRun[iY][iR],false);
		    insertCell(lMapRun,hMap[iY][iR],iBX+iNeigh,iBY-jNeigh,lumiRun[iY][iR],false);
		    insertCell(lMapRun,hMap[iY][iR],iBX+iNeigh,iBY+jNeigh,lumiRun[iY][iR],false);
		  }
		}
		
		
	      }
	      
	      
	    }
	    
	  }//loop on binsY
	}//loop on binsX

	TFile *fTmp = 0;
	if (!doCold) fTmp = TFile::Open(("hotjets-"+yearShort[iY]+"run"+runPeriod[iY][iR]+"_"+DR[idr]+".root").c_str(),"RECREATE");
	else fTmp = TFile::Open(("coldjets-"+yearShort[iY]+"run"+runPeriod[iY][iR]+"_"+DR[idr]+".root").c_str(),"RECREATE");
	TH2D *h2Drun = do17?(TH2D*)hMap[0][0]->Clone(("map"+DR[idr]).c_str()):(TH2D*)hMap[1][0]->Clone(("map"+DR[idr]).c_str());
	h2Drun->Reset();
	
	std::map<std::pair<double,double>,double>::iterator lIter = lMapRun.begin();
	std::cout << " -- Found " << lMapRun.size() << " entries in run map" << std::endl;
	for ( ; lIter != lMapRun.end(); ++lIter){
	  h2Drun->Fill(lIter->first.first,lIter->first.second,1.);
	  
	  std::pair<std::map<std::pair<double,double>,double>::iterator,bool> lInsert = lMap.insert(std::pair<std::pair<double,double>,double>(lIter->first,lumiRun[iY][iR]));
	  if (!lInsert.second) {
	    double & lTmp = lInsert.first->second;
	    //if (lTmp.find(year[iY])==lTmp.npos) lTmp += year[iY];
	    //lTmp += run[iY][iR];
	    lTmp += lumiRun[iY][iR];
	  }
	}
	fTmp->cd();
	h2Drun->Write();
	fTmp->Write();
	fTmp->Close();
	
      }//loop on runs
    }//loop on years
    std::map<std::pair<double,double>,double>::iterator lIter = lMap.begin();
    std::cout << " --------------------------------"
	      << " -- Summary ---------------------"
	      << " --------------------------------"
	      << std::endl;
    
    TH2D *h2D = do17?(TH2D*)hMap[0][0]->Clone("all"):(TH2D*)hMap[1][0]->Clone("all");
    h2D->Reset();
    
    for ( ; lIter != lMap.end(); ++lIter){
      //unsigned idx = convertToInt(lIter->second);
      //std::cout << lIter->first.first << " " << lIter->first.second << " " << lIter->second << " " << idx << std::endl;
      h2D->Fill(lIter->first.first,lIter->first.second,do17 && !do18?lIter->second/lumi17:do18 && !do17?lIter->second/lumi18:lIter->second/lumiTot);
    }
    
    myc2->cd();
    
    std::string label = doCold?"cold":"hot";
    gStyle->SetPalette(1);
    h2D->SetTitle(";#eta;#phi;Lumi Fraction");
    h2D->Draw("colz");
    if (do17 && !do18) lat.DrawLatexNDC(0.2,0.96,("2017 "+DR[idr]+" "+label+" map").c_str());
    else if (!do17 && do18) lat.DrawLatexNDC(0.2,0.96,("2018 "+DR[idr]+" "+label+" map").c_str());
    else if (do17 && do18) lat.DrawLatexNDC(0.2,0.96,("2017+2018 "+DR[idr]+" "+label+" map").c_str());
    gPad->SetRightMargin(0.15);
    gPad->SetGridx(1);
    gPad->SetGridy(1);
    
    gPad->Update();
    gPad->Print(do17 && !do18?(label+"Towers_"+DR[idr]+"_2017.pdf").c_str():do18 && !do17 ? (label+"Towers_"+DR[idr]+"_2018.pdf").c_str():(label+"Towers_"+DR[idr]+"_allYears.pdf").c_str());
    gPad->Print(do17 && !do18?(label+"Towers_"+DR[idr]+"_2017.png").c_str():do18 && !do17 ? (label+"Towers_"+DR[idr]+"_2018.png").c_str():(label+"Towers_"+DR[idr]+"_allYears.png").c_str());
    gPad->SetLogz(1);
    h2D->GetZaxis()->SetRangeUser(0.01,1);
    gPad->Update();
    gPad->Print(do17 && !do18?(label+"Towers_"+DR[idr]+"_2017_logz.pdf").c_str():do18 && !do17? (label+"Towers_"+DR[idr]+"_2018_logz.pdf").c_str():(label+"Towers_"+DR[idr]+"_allYears_logz.pdf").c_str());
    gPad->Print(do17 && !do18?(label+"Towers_"+DR[idr]+"_2017_logz.png").c_str():do18 && !do17? (label+"Towers_"+DR[idr]+"_2018_logz.png").c_str():(label+"Towers_"+DR[idr]+"_allYears_logz.png").c_str());
  
    TFile *fout = new TFile(do17 && !do18?(label+"Towers_"+DR[idr]+"_2017.root").c_str():do18 && !do17 ? (label+"Towers_"+DR[idr]+"_2018.root").c_str():(label+"Towers_"+DR[idr]+"_allYears.root").c_str(),"RECREATE");
    fout->cd();
    h2D->Write();
    fout->Write();

  }//loop on DR values
  
  return 0;
}//main
