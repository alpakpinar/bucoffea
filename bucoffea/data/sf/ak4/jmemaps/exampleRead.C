#include <iostream>
#include <sstream>
#include <fstream>

#include "TCanvas.h"
#include "TPad.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TLatex.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TLatex.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TGaxis.h"
#include "TString.h"

double getHotMapWeight(const bool isData,
		       TH2D* hMap,
		       const double & eta, const double & phi){
  int binX = hMap->GetXaxis()->FindBin(eta);
  int binY = hMap->GetYaxis()->FindBin(phi);
  double tmpW = hMap->GetBinContent(binX,binY);
  if (isData && tmpW>0) return 0;
  else if (tmpW>0) return (1-tmpW);
  return 1.;
};

int exampleRead(){
  const unsigned nM = 2;
  std::string mapName[nM] = {"hot","cold"};

  const unsigned nDR = 2;
  std::string DR[nDR] = {"dR0","dR2"};
  
  const unsigned nRP = 4;
  std::string runPeriod[2][4] = {
    {"B","C","DE","F"},
    {"A","B","C","D"}
  };

  //cold and hot maps
  TFile *fMap[nM][nDR][nRP];
  TH2D *hMap[nM][nDR][nRP];
  if (!isData) {
    for (unsigned idr(0); idr<nDR; ++idr){
      for (unsigned iM(0); iM<nM; ++iM){
        fMap[iM][idr][0] = TFile::Open((mapName[iM]+"Towers_"+DR[idr]+"_"+year[iY]+".root").c_str());
        if (!fMap[iM][idr][0]) return 1;
        hMap[iM][idr][0] = (TH2D*)gDirectory->Get("all");
        if (!hMap[iM][idr][0]) return 1;
      }
    }
  }
  else {
    for (unsigned iRP(0); iRP<nRP; ++iRP){//loop on run
      std::cout << " -- run " << runPeriod[iY][iRP] << std::endl;
      for (unsigned idr(0); idr<nDR; ++idr){
        for (unsigned iM(0); iM<nM; ++iM){
          fMap[iM][idr][iRP] = TFile::Open((mapName[iM]+"jets-"+yearShort[iY]+"run"+runPeriod[iY][iRP]+"_"+DR[idr]+".root").c_str());
          if (!fMap[iM][idr][iRP]) return 1;
          fMap[iM][idr][iRP]->cd();
          hMap[iM][idr][iRP] = (TH2D*)gDirectory->Get(("map"+DR[idr]).c_str());
          if (!hMap[iM][idr][iRP]) return 1;
        }
      }
    }
  }
  
	
  double mapW[nM][nDR];
  TBranch *brMapdR[nM][nDR];
  for (unsigned iM(0); iM<nM; ++iM){
    for (unsigned idr(0); idr<nDR; ++idr){
      mapW[iM][idr] = 0;
      brMapdR[iM][idr] = 0;
    }
  }

  
  for (unsigned iM(0); iM<nM; ++iM){
    for (unsigned idr(0); idr<nDR; ++idr){
      brMapdR[iM][idr] = treeOut->Branch((mapName[iM]+"MapWeight"+DR[idr]).c_str(),&mapW[iM][idr]);
    }
  }
  
    
  for (unsigned iM(0); iM<nM; ++iM){
    for (unsigned idr(0); idr<nDR; ++idr){
      
      double w1 = getHotMapWeight(isData,hMap[iM][idr][iRP],eta[0],phi[0]);
      double w2 = getHotMapWeight(isData,hMap[iM][idr][iRP],eta[1],phi[1]);
      mapW[iM][idr] = w1*w2;
      //if (!startFromSkim)
      brMapdR[iM][idr]->Fill();
      
    }
  }
  
  return 0;
}//main
