package org.opencv.samples.facedetect;

import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.Toast;


public class CustomOnItemSelectedListener implements OnItemSelectedListener {
	//public static String Selection;
	
	public void onItemSelected(AdapterView parent, View view, int pos,long id) {
	  //Selection = parent.getItemAtPosition(pos).toString();
	  Toast.makeText(parent.getContext(), 
		"Processing selected: " + parent.getItemAtPosition(pos).toString(),
		Toast.LENGTH_SHORT).show();
	}
 
  @Override
  	public void onNothingSelected(AdapterView arg0) {
	// TODO Auto-generated method stub
  	}
 
}