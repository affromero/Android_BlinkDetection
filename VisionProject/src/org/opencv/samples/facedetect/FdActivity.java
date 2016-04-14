package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.renderscript.Long2;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

public class FdActivity extends Activity implements CvCameraViewListener2 {

	private static final String TAG = "OCVSample::Activity";
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
	public static final int JAVA_DETECTOR = 0;
	
	Core.MinMaxLocResult mmG;
	Rect eye_only_rectangle;
	Point iris;
	Rect eye_template;

	private int cameraid = 1;
	private Mat templateR;
	private Mat templateL;
	private Mat templateR_open;
	private Mat templateL_open;
	
	private boolean HaarLE = false;
	private boolean HaarRE = false;
	private boolean HaarEyeOpen_R = false;
	private boolean HaarEyeOpen_L = false;
	
	private MenuItem mItemFace50;
	private MenuItem mItemFace40;
	private MenuItem mItemFace30;
	private MenuItem mItemFace20;
	private MenuItem mItemType;

	private Mat mRgba;
	private Mat mGray;

	private File mCascadeFile;
	private File cascadeFileER;
	private File cascadeFileEL;
	private File cascadeFileEyeOpen;
	
	private CascadeClassifier mJavaDetector;
	private CascadeClassifier mJavaDetectorEyeRight;
	private CascadeClassifier mJavaDetectorEyeLeft;
	private CascadeClassifier mJavaDetectorEyeOpen;
	
	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSize = 0;

	private CameraBridgeViewBase mOpenCvCameraView;
	
	int AllTime = 30;
	int drowsyTime = 1;
	double frequency;
	long timer;
	int TotalFrames = 0;
	int FrameFace = 0;
	int FrameEyesOpen = 0;
	int FrameEyesClosed = 0;
	public int FrameClosedDrowsy = 0;
	boolean flag = false;
	boolean flag_drowsy = false;
	boolean drowsy = true;
	long timer_drowsy;
    int count_drowsy = 0;
	MediaPlayer beep;
	
    
//
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				try {
					// load cascade file from application resources
					//Face detection classifier
					InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
					mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
					FileOutputStream os = new FileOutputStream(mCascadeFile);

					byte[] buffer = new byte[4096];
					int bytesRead;
					while ((bytesRead = is.read(buffer)) != -1) {
						os.write(buffer, 0, bytesRead);
					}
					is.close();
					os.close();
					
					// ------------------ load right eye classificator -----------------------
					InputStream iser = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
					File cascadeDirER = getDir("cascadeER",Context.MODE_PRIVATE);
					cascadeFileER = new File(cascadeDirER,"haarcascade_eye_right.xml");
					FileOutputStream oser = new FileOutputStream(cascadeFileER);
					
					byte[] bufferER = new byte[4096];
					int bytesReadER;
					while ((bytesReadER = iser.read(bufferER)) != -1) {
						oser.write(bufferER, 0, bytesReadER);
					}
					iser.close();
					oser.close();

					// ------------------ load left eye classificator -----------------------
					InputStream isel = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
					File cascadeDirEL = getDir("cascadeEL",Context.MODE_PRIVATE);
					cascadeFileEL = new File(cascadeDirEL,"haarcascade_eye_left.xml");
					FileOutputStream osel = new FileOutputStream(cascadeFileEL);
					
					byte[] bufferEL = new byte[4096];
					int bytesReadEL;
					while ((bytesReadEL = isel.read(bufferEL)) != -1) {
						osel.write(bufferEL, 0, bytesReadEL);
					}
					isel.close();
					osel.close();
					
					// ------------------ load open eye classificator -----------------------
					InputStream opisel = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
					File cascadeDirEyeOpen = getDir("cascadeEyeOpen",Context.MODE_PRIVATE);
					cascadeFileEyeOpen = new File(cascadeDirEyeOpen,"haarcascade_eye_tree_eyeglasses.xml");
					FileOutputStream oposel = new FileOutputStream(cascadeFileEyeOpen);
					
					byte[] bufferEyeOpen = new byte[4096];
					int bytesReadEyeOpen;
					while ((bytesReadEyeOpen = opisel.read(bufferEyeOpen)) != -1) {
						oposel.write(bufferEyeOpen, 0, bytesReadEyeOpen);
					}
					opisel.close();
					oposel.close();
					
					//Face Classifier 
					mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
					if (mJavaDetector.empty()) {
						Log.e(TAG, "Failed to load cascade classifier of face");
						mJavaDetector = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "+ mCascadeFile.getAbsolutePath());
					//cascadeDir.delete();
					
					//EyeRightClassifier
					mJavaDetectorEyeRight = new CascadeClassifier(cascadeFileER.getAbsolutePath());
					if (mJavaDetectorEyeRight.empty()) {
						Log.e(TAG, "Failed to load cascade classifier of eye right");
						mJavaDetectorEyeRight = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileER.getAbsolutePath());
					//cascadeDirER.delete();

					//EyeLeftClassifier
					mJavaDetectorEyeLeft = new CascadeClassifier(cascadeFileEL.getAbsolutePath());
					if (mJavaDetectorEyeLeft.empty()) {
						Log.e(TAG, "Failed to load cascade classifier of eye left");
						mJavaDetectorEyeLeft = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEL.getAbsolutePath());
					//cascadeDirEL.delete();
					
					//EyeOpenClassifier
					mJavaDetectorEyeOpen = new CascadeClassifier(cascadeFileEyeOpen.getAbsolutePath());
					if (mJavaDetectorEyeOpen.empty()) {
						Log.e(TAG, "Failed to load cascade classifier of eye open");
						mJavaDetectorEyeOpen = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEyeOpen.getAbsolutePath());
					//cascadeDirEyeOpen.delete();
					
				} catch (IOException e) {
					e.printStackTrace();
					Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
				}
				mOpenCvCameraView.setCameraIndex(cameraid);
				mOpenCvCameraView.enableFpsMeter();
				mOpenCvCameraView.enableView();
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	public FdActivity() {
		mDetectorName = new String[2];
		mDetectorName[JAVA_DETECTOR] = "Java";
		Log.i(TAG, "Instantiated new " + this.getClass());
	}
	
	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
		
		beep = MediaPlayer.create(this, R.raw.button1);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
		System.exit(0);
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,mLoaderCallback);
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
		//System.exit(0);
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		if (drowsy){
			timer_drowsy = Core.getTickCount();
			drowsy = false;
		}
			
		SetTimer();
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		TotalFrames++;
		
		boolean showing_drowsy = SetDrowsy();
		if (showing_drowsy || count_drowsy != 0){
			count_drowsy++;
			Core.putText(mRgba, "ALERT!", new Point(mRgba.size().width/2, mRgba.size().height/2), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 4, new Scalar(255,255,0),5);
			if (count_drowsy>2){count_drowsy=0;}
		}
		
		
		if (mAbsoluteFaceSize == 0) {
			int height = mGray.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
			}
		}

		MatOfRect faces = new MatOfRect();

		if (mJavaDetector != null)
			//detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
			mJavaDetector.detectMultiScale(mGray, //Input image over perform classifier with
					faces, //List of rectangles where are found whatever needs to classifier.
					1.1, //Scalefactor. How much the image is reduced at each image scale
					2,    //MinNeighbors. Specify how many neighbors each candidate rectangle should have to retain it. 
					2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
					new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),	//Minimum possible object size. Objects smaller than that are ignored.
					new Size()
			);  //Maximum possible object size. Objects larger than that are ignored.

		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++) {
			
			//Draw a rectangle on mRgba, from point top-left of faces found to bottom right, color: FACE_RECT_COLOR, lineWidth: 3
			Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),FACE_RECT_COLOR, 3);
			
			//Rectangle of the face
			Rect RectOfFace = facesArray[i];
			//Split two different regions for two eyes			
			///*
			Rect eyearea_right = new Rect( RectOfFace.x + RectOfFace.width / 16 , 
					(int) (RectOfFace.y + (RectOfFace.height / 4.5)) ,
	                (RectOfFace.width - 2 * RectOfFace.width / 16) / 2, 
	                (int) (RectOfFace.height / 3.0)
	                );
			
	        Rect eyearea_left = new Rect( RectOfFace.x + RectOfFace.width / 16 + ( RectOfFace.width - 2 * RectOfFace.width / 16 ) / 2 ,
	                 (int) (RectOfFace.y + (RectOfFace.height / 4.5)) , 
	                 (RectOfFace.width - 2 * RectOfFace.width / 16) / 2 , 
	                 (int) (RectOfFace.height / 3.0)
	                 );
	        
            //*/
            //If you want to watch sub-matrix on display uncomment next and uncomment before
            /*
		    mRgba = mRgba.submat(RectOfFace);
			Imgproc.resize(mRgba, mRgba, mGray.size());
			//Split two different regions for two eyes			
			
			Rect eyearea_right = new Rect( mRgba.cols() / 16 , 
					(int) ((mRgba.rows() / 4.5)) ,
	                (mRgba.cols() - 2 * mRgba.cols() / 16) / 2, 
	                (int) (mRgba.rows() / 3.0)
	                );
			
	        Rect eyearea_left = new Rect(  mRgba.cols() / 16 + (  mRgba.cols() - 2 *  mRgba.cols() / 16 ) / 2 ,
	                 (int) ((mRgba.rows() / 4.5)) , 
	                 ( mRgba.cols() - 2 *  mRgba.cols() / 16) / 2 , 
	                 (int) (mRgba.rows() / 3.0)
	                 );
	        
            Core.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                    new Scalar(255, 0, 0, 255), 2);
            Core.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                    new Scalar(255, 0, 0, 255), 2);
            */
            
			FrameFace++;
			//get_template function needs: classifier, area over perform classifier, and desired size of new template
			Rect rectR = get_template(mJavaDetectorEyeRight, eyearea_right);
			
			Rect rectL = get_template(mJavaDetectorEyeLeft, eyearea_left);
			if (rectL.width==0 || rectL.height==0 || rectR.width==0 || rectR.height==0){continue;}
			

			
			rectR = get_template(mJavaDetectorEyeOpen, rectR, new Size(1, 1), new Size(50,50));
			templateR_open = mGray.submat(rectR);
			
			rectL = get_template(mJavaDetectorEyeOpen, rectL, new Size(1, 1), new Size(50,50));
			templateL_open = mGray.submat(rectL);

			/*
			if (rectL.width>0){
			    mRgba = mRgba.submat(rectR);
				Imgproc.resize(mRgba, mRgba, mGray.size());
			}
			*/
			
			//match_eye 
			HaarEyeOpen_R = match_eye(templateR_open); 
			HaarEyeOpen_L = match_eye(templateL_open); 
			
			if(!HaarEyeOpen_R && !HaarEyeOpen_L){
				Core.putText(mRgba, "Closed", new Point(mRgba.size().width/18, mRgba.size().height/5), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 4, new Scalar(0,255,0),5);
				FrameEyesClosed++;
				FrameClosedDrowsy++;
			}
			else if (HaarEyeOpen_R && HaarEyeOpen_L){
				Core.putText(mRgba, "Open", new Point(mRgba.size().width/18, mRgba.size().height/5), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 4, new Scalar(0,255,0),5);
				FrameEyesOpen++;
			}
			
			break;
		}
		return mRgba;
	}
	
	private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest) {
		Mat template = new Mat(); //Where is gonna be stored the eye detected data
		Mat mROI = mGray.submat(RectAreaInterest); //Matrix which contain data of the whole eye area from geometry of face
		MatOfRect eyes = new MatOfRect(); 
		iris = new Point();
		eye_template = new Rect();
		//detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
		clasificator.detectMultiScale(mROI, //Image which set classification. Needs to be of the type CV_8U
				eyes, //List of rectangles where are stored possibles eyes detected 
				1.1, //Scalefactor. How much the image is reduced at each image scale
				2,    //MinNeighbors. Specify how many neighbors each candidate rectangle should have to retain it. 
				Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, //0 or 1.
				new Size(10, 10), //Minimum possible object size. Objects smaller than that are ignored.
				new Size(100,100)        //Maximum possible object size. Objects larger than that are ignored.
		);

		Rect[] eyesArray = eyes.toArray();
		for (int i = 0; i < eyesArray.length;) {
			Rect eyeDetected = eyesArray[i];
			eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
			eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

			mROI = mGray.submat(eyeDetected);
			mmG = Core.minMaxLoc(mROI);
			
			iris.x = mmG.minLoc.x + eyeDetected.x;
			iris.y = mmG.minLoc.y + eyeDetected.y;
			eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

			//Imgproc.equalizeHist(template, template);
			break;
			//return template;
		}
		return eye_template;
	}
	

	private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest, Size min_size, Size max_size) {
		Mat template = new Mat(); //Where is gonna be stored the eye detected data
		Mat mROI = mGray.submat(RectAreaInterest); //Matrix which contain data of the whole eye area from geometry of face
		MatOfRect eyes = new MatOfRect(); 
		iris = new Point();
		eye_template = new Rect();
		//detectMultiScale(const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3, int flags=0, Size minSize=Size(), Size maxSize=Size())
		clasificator.detectMultiScale(mROI, //Image which set classification. Needs to be of the type CV_8U
				eyes, //List of rectangles where are stored possibles eyes detected 
				1.01, //Scalefactor. How much the image is reduced at each image scale
				2,    //MinNeighbors. Specify how many neighbors each candidate rectangle should have to retain it. 
				Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, //0 or 1.
				min_size, //Minimum possible object size. Objects smaller than that are ignored.
				max_size        //Maximum possible object size. Objects larger than that are ignored.
		);

		Rect[] eyesArray = eyes.toArray();
		for (int i = 0; i < eyesArray.length;) {
			Rect eyeDetected = eyesArray[i];
			eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
			eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

			mROI = mGray.submat(eyeDetected);
			mmG = Core.minMaxLoc(mROI);
			
			iris.x = mmG.minLoc.x + eyeDetected.x;
			iris.y = mmG.minLoc.y + eyeDetected.y;
			eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

			//Imgproc.equalizeHist(template, template);
			break;

			//return template;
		}
		return eye_template;
	}
	
	private boolean match_eye(Mat mTemplate) {
		 //Check for bad template size
		if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
			return false;
		}else{
			return true;
		}
	}

	
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemFace50 = menu.add("Face size 50%");
		mItemFace40 = menu.add("Face size 40%");
		mItemFace30 = menu.add("Face size 30%");
		mItemFace20 = menu.add("Face size 20%");
		mItemType = menu.add(mDetectorName[mDetectorType]);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item == mItemFace50)
			setMinFaceSize(0.5f);
		else if (item == mItemFace40)
			setMinFaceSize(0.4f);
		else if (item == mItemFace30)
			setMinFaceSize(0.3f);
		else if (item == mItemFace20)
			setMinFaceSize(0.2f);
		else if (item == mItemType) {
			int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
			item.setTitle(mDetectorName[tmpDetectorType]);
		}
		return true;
	}

	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}
	
	public void onToggleClick(View v) {
		cameraid = cameraid^1;
		mOpenCvCameraView.disableView();
	    mOpenCvCameraView.setCameraIndex(cameraid);
	    mOpenCvCameraView.enableView();
    }
	
	public void InitTimer(View v){
		Toast.makeText(getApplicationContext(), "Timer enabled for "+AllTime+" seconds", Toast.LENGTH_SHORT).show();
		frequency = Core.getTickFrequency(); //frecuency of the clock. How many clocks cycles per second,
		timer = Core.getTickCount();			//start timer for 1 minute. It gives number of clock cycles.
		TotalFrames = 0;
		FrameFace = 0;
		FrameEyesOpen = 0;
		FrameEyesClosed = 0;
		flag = true;
	}
	
	public void SetTimer(){
		long newtimer = Core.getTickCount()-timer;
		if(newtimer/frequency>AllTime && flag){
			if(FrameEyesClosed>FrameFace){FrameEyesClosed=FrameFace;}
			if(FrameEyesOpen>FrameFace){FrameEyesOpen=FrameFace;}
			AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
			audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, 10, 0);
			beep.start();
			String msg = "Timer: "+newtimer+" Frecuency: "+(long)frequency;
			final String Result = "Total Frames: "+TotalFrames+"\nFrames face: "+FrameFace+
					"\nFrames EyesOpen: "+FrameEyesOpen+"\nFrames EyesClosed: "+FrameEyesClosed;
			Log.i(TAG, msg);
			Log.i(TAG, Result);
			
			runOnUiThread(new Runnable() { //Toast crashes when is used gettickcount. So that it is needed
		            public void run(){
		            	Toast.makeText(getApplicationContext(), Result, Toast.LENGTH_LONG).show();
		            }
		        });
			flag = false;
		}
	}
	
	public boolean SetDrowsy(){
		long newtimer = Core.getTickCount()-timer_drowsy;
		frequency = Core.getTickFrequency();
		flag_drowsy = false;
		if(newtimer/frequency>drowsyTime){
			timer_drowsy = Core.getTickCount();
			if (FrameClosedDrowsy>2){
				flag_drowsy = true;
			}
			FrameClosedDrowsy = 0;
		}
		return flag_drowsy;
	}
}
