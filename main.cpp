#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat frame);

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

static vector<string> cascadeSearchDirs() {
    vector<string> dirs;
    if (const char* env = std::getenv("OPENCV_CASCADES_PATH"))      dirs.emplace_back(env);
    if (const char* env2 = std::getenv("OPENCV_SAMPLES_DATA_PATH")) dirs.emplace_back(env2); // suele ser .../share/opencv4
    // Rutas comunes (macOS + Homebrew/MacPorts + Linux)
    dirs.emplace_back("/opt/homebrew/opt/opencv/share/opencv4/haarcascades"); // macOS ARM (M1/M2/M3/M4)
    dirs.emplace_back("/usr/local/opt/opencv/share/opencv4/haarcascades");    // macOS Intel
    dirs.emplace_back("/opt/local/share/opencv4/haarcascades");               // MacPorts
    dirs.emplace_back("/usr/share/opencv4/haarcascades");                     // Linux típ.
    // Repo/local
    dirs.emplace_back("haarcascades");
    dirs.emplace_back("data/haarcascades");
    return dirs;
}

static bool loadCascadeAuto(CascadeClassifier& cc, const string& fileName, string& usedPath) {
    // 1) Si OPENCV_SAMPLES_DATA_PATH apunta a .../share/opencv4, intenta via samples::findFile
    try {
        string found = samples::findFile("haarcascades/" + fileName, false);
        if (!found.empty() && cc.load(found)) { usedPath = found; return true; }
    } catch (...) { /* ignorar */ }

    // 2) Probar en rutas conocidas
    for (const auto& dir : cascadeSearchDirs()) {
        string full = dir;
        if (!full.empty() && full.back() != '/') full.push_back('/');
        full += fileName;
        if (cc.load(full)) { usedPath = full; return true; }
    }

    // 3) Último intento: archivo en el cwd
    if (cc.load(fileName)) { usedPath = fileName; return true; }

    return false;
}

int main() {
    const string FACE_FILE = "haarcascade_frontalface_alt.xml";
    const string EYES_FILE = "haarcascade_eye_tree_eyeglasses.xml";

    string facePath, eyesPath;

    cout << "Cargando cascadas...\n";
    if (!loadCascadeAuto(face_cascade, FACE_FILE, facePath)) {
        cerr << "--(!) No se pudo cargar la cascade de rostro: " << FACE_FILE << "\n";
        cerr << "   Revisa que OpenCV esté instalado (Homebrew) y que existan los XML.\n";
        return -1;
    }
    if (!loadCascadeAuto(eyes_cascade, EYES_FILE, eyesPath)) {
        cerr << "--(!) No se pudo cargar la cascade de ojos: " << EYES_FILE << "\n";
        return -1;
    }

    cout << "Usando face cascade: " << facePath << "\n";
    cout << "Usando eyes cascade: " << eyesPath << "\n";

    int camera_device = 0; // por defecto, cámara 0
    VideoCapture capture;
    capture.open(camera_device);
    if (!capture.isOpened()) {
        cerr << "--(!) Error abriendo la cámara (" << camera_device << ")\n";
        cerr << "   macOS: revisa Permisos en Sistema > Privacidad y seguridad > Cámara.\n";
        return -1;
    }

    Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            cerr << "--(!) Frame vacío -- Break!\n";
            break;
        }
        detectAndDisplay(frame);
        if (waitKey(10) == 27) break; // ESC
    }
    return 0;
}

void detectAndDisplay(Mat frame) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detectar rostros
    vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++) {
        Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
        //cout<<center<<"\n";
        Point p1(faces[i].x, faces[i].y);
        Point p2(faces[i].x + faces[i].width, faces[i].y+faces[i].height);

        rectangle(frame, p1, p2, Scalar(255, 0, 255), 4);
        //ellipse(frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 4);
        Mat faceROI = frame_gray(faces[i]);

        // Detectar ojos en cada rostro
        /*vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++) {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2,
                             faces[i].y + eyes[j].y + eyes[j].height/2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
        }*/
    }

    imshow("Capture - Face detection", frame);
}