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

enum class TrackState { DETECT, TRACK };
TrackState gState = TrackState::DETECT;

cv::Mat gTemplateT;
bool gHaveTemplate = false;
const cv::Size kTemplSize(64, 64);
double gMatchThr = 0.65;
int gMissCount = 0;
int gMissLimit = 5;

static vector<string> cascadeSearchDirs() {
    vector<string> dirs;
    if (const char* env = std::getenv("OPENCV_CASCADES_PATH"))      dirs.emplace_back(env);
    if (const char* env2 = std::getenv("OPENCV_SAMPLES_DATA_PATH")) dirs.emplace_back(env2); // suele ser .../share/opencv4

    dirs.emplace_back("/opt/homebrew/opt/opencv/share/opencv4/haarcascades"); // macOS ARM
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

// Construye/actualiza el template T a partir de un rostro detectado
static bool buildTemplateFromFaceGray(const cv::Mat& frame_gray_eq, const cv::Rect& faceRect) {
    cv::Rect roi = faceRect & cv::Rect(0,0, frame_gray_eq.cols, frame_gray_eq.rows);
    if (roi.empty()) return false;

    // Tomamos directamente la subimagen del gris ecualizado
    gTemplateT = frame_gray_eq(roi).clone();     // CV_8U
    cv::resize(gTemplateT, gTemplateT, kTemplSize);
    gTemplateT.convertTo(gTemplateT, CV_32F, 1.0/255.0); // normalizar 0..1

    gHaveTemplate = true;
    gMissCount = 0;
    return true;
}

// Intenta localizar T en el frame actual
static bool matchTemplateOnce(const cv::Mat& frameGray, cv::Rect& foundBox, double& score) {
    if (!gHaveTemplate || gTemplateT.empty()) return false;
    if (frameGray.cols < gTemplateT.cols || frameGray.rows < gTemplateT.rows) return false;

    cv::Mat f32;
    frameGray.convertTo(f32, CV_32F, 1.0/255.0);

    cv::Mat result;
    cv::matchTemplate(f32, gTemplateT, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal; cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    score = maxVal;

    cv::putText(const_cast<cv::Mat&>(frameGray), cv::format("max %.2f", score),
                {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, 255, 2);

    if (maxVal >= gMatchThr) {
        foundBox = cv::Rect(maxLoc.x, maxLoc.y, gTemplateT.cols, gTemplateT.rows);
        return true;
    }
    return false;
}

int main() {
    const string FACE_FILE = "haarcascade_frontalface_default.xml";

    string facePath;

    cout << "Cargando cascadas...\n";
    if (!loadCascadeAuto(face_cascade, FACE_FILE, facePath)) {
        cerr << "--(!) No se pudo cargar la cascade de rostro: " << FACE_FILE << "\n";
        cerr << "   Revisa que OpenCV esté instalado (Homebrew) y que existan los XML.\n";
        return -1;
    }
    cout << "Usando face cascade: " << facePath << "\n";

    const string videoPath = "/Users/hectorperez/Downloads/face_cascade_classifier_block_matching-main-2/walkingppl.mp4"; // usa ruta absoluta si es necesario
    VideoCapture capture;
    if (!capture.open(videoPath, cv::CAP_ANY)) {
        cerr << "--(!) No se pudo abrir el video: " << videoPath << "\n";
        cerr << "   Prueba con la ruta absoluta o convierte a MP4/H.264.\n";
        return -1;
    }

    Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            cerr << "--(!) Frame vacío -- Break!\n";
            break;
        }
        detectAndDisplay(frame);

        int delay = 1;
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps > 1.0) delay = max(1, (int)std::round(1000.0 / fps));
        int key = waitKey(delay);
        if (key == 27) break;             // ESC para salir
        if (key == 'r' || key == 'R') {   // 'r' para reiniciar el video
            capture.set(cv::CAP_PROP_POS_FRAMES, 0);
        }
    }
    return 0;
}

void detectAndDisplay(Mat frame) {
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    switch (gState) {
    case TrackState::TRACK: {
        cv::Rect box; double score = 0.0;
        if (matchTemplateOnce(frame_gray, box, score)) {

            rectangle(frame, box, Scalar(0,255,0), 3);
            putText(frame, cv::format("match %.2f", score),
                    {box.x, std::max(0, box.y-5)}, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);

            // static int frameCount = 0; frameCount++;
            // if ((frameCount % 15) == 0 || score < 0.88) buildTemplateFromFace(frame, box);

            gMissCount = 0;
        } else {
            putText(frame, cv::format("no match (%.2f)", score),
            {10, 30}, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2);
            gMissCount++;
            if (gMissCount >= gMissLimit) {
                gState = TrackState::DETECT;
                gHaveTemplate = false;
            }
        }
        break;
    }

    case TrackState::DETECT: {
        vector<Rect> faces;
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0, Size(60,60));

        if (!faces.empty()) {
            rectangle(frame, faces[0], Scalar(255, 0, 255), 3);

            if (buildTemplateFromFaceGray(frame_gray, faces[0])) {
                putText(frame, "Template init", {faces[0].x, max(0, faces[0].y-8)},
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,0,255), 2);
                gState = TrackState::TRACK;
            }
        }
        break;
    }
    }

    imshow("Capture - Face detection", frame);
}