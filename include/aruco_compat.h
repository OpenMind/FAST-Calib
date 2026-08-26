/*
Compatibility shim for the two OpenCV ArUco APIs.

OpenCV >= 4.7 moved ArUco from the contrib module `opencv2/aruco.hpp` into
`opencv2/objdetect`, replaced the free detection/pose functions with the
`ArucoDetector` class, and made Dictionary/Board/DetectorParameters value
types. JetPack's OpenCV 4.8 ships only the new API, upstream FAST-Calib was
written against the old one.

This header exposes one set of names (namespace `fc_aruco`) that behaves the
same on both, so qr_detect.hpp does not need version branches inline.
*/

#ifndef ARUCO_COMPAT_H
#define ARUCO_COMPAT_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
  #define FAST_CALIB_ARUCO_OBJDETECT 1
  #include <opencv2/objdetect/aruco_detector.hpp>
#else
  #define FAST_CALIB_ARUCO_OBJDETECT 0
  #include <opencv2/aruco.hpp>
#endif

namespace fc_aruco
{

#if FAST_CALIB_ARUCO_OBJDETECT

using Dictionary = cv::aruco::Dictionary;
using Board = cv::aruco::Board;
using DetectorParameters = cv::aruco::DetectorParameters;

inline Dictionary getPredefinedDictionary(int dict)
{
  return cv::aruco::getPredefinedDictionary(dict);
}

inline Board makeBoard(const std::vector<std::vector<cv::Point3f>> &obj_points,
                       const Dictionary &dictionary, const std::vector<int> &ids)
{
  return Board(obj_points, dictionary, ids);
}

inline DetectorParameters makeDetectorParameters() { return DetectorParameters(); }

inline void setCornerRefineSubpix(DetectorParameters &params)
{
  params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
}

inline void detectMarkers(cv::InputArray image, const Dictionary &dictionary,
                          std::vector<std::vector<cv::Point2f>> &corners,
                          std::vector<int> &ids, const DetectorParameters &params)
{
  cv::aruco::ArucoDetector detector(dictionary, params);
  detector.detectMarkers(image, corners, ids);
}

// Replacement for the removed cv::aruco::estimatePoseSingleMarkers(): solve one
// IPPE_SQUARE PnP per marker against its canonical corner layout (the same
// corner order detectMarkers() returns: top-left, top-right, bottom-right,
// bottom-left, in a marker frame centred on the marker with +z out of it).
inline void estimatePoseSingleMarkers(const std::vector<std::vector<cv::Point2f>> &corners,
                                      float marker_length, cv::InputArray camera_matrix,
                                      cv::InputArray dist_coeffs,
                                      std::vector<cv::Vec3d> &rvecs,
                                      std::vector<cv::Vec3d> &tvecs)
{
  const float h = marker_length / 2.f;
  const std::vector<cv::Point3f> marker_object_points{
      {-h, h, 0.f}, {h, h, 0.f}, {h, -h, 0.f}, {-h, -h, 0.f}};

  rvecs.resize(corners.size());
  tvecs.resize(corners.size());
  for (size_t i = 0; i < corners.size(); ++i)
  {
    cv::solvePnP(marker_object_points, corners[i], camera_matrix, dist_coeffs,
                 rvecs[i], tvecs[i], false, cv::SOLVEPNP_IPPE_SQUARE);
  }
}

// Replacement for the removed cv::aruco::estimatePoseBoard(): match the detected
// corners to the board's 3D points and solve a single PnP over all of them.
// Returns the number of markers used, 0 when the pose could not be estimated.
inline int estimatePoseBoard(const std::vector<std::vector<cv::Point2f>> &corners,
                             const std::vector<int> &ids, const Board &board,
                             cv::InputArray camera_matrix, cv::InputArray dist_coeffs,
                             cv::Vec3d &rvec, cv::Vec3d &tvec,
                             bool use_extrinsic_guess = false)
{
  cv::Mat object_points, image_points;
  board.matchImagePoints(corners, ids, object_points, image_points);
  if (object_points.total() < 4) return 0;

  if (!cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec,
                    tvec, use_extrinsic_guess))
    return 0;

  return static_cast<int>(object_points.total() / 4);
}

#else  // pre-4.7 contrib API: same names, Ptr-based types

using Dictionary = cv::Ptr<cv::aruco::Dictionary>;
using Board = cv::Ptr<cv::aruco::Board>;
using DetectorParameters = cv::Ptr<cv::aruco::DetectorParameters>;

inline Dictionary getPredefinedDictionary(int dict)
{
  return cv::aruco::getPredefinedDictionary(
      static_cast<cv::aruco::PREDEFINED_DICTIONARY_NAME>(dict));
}

inline Board makeBoard(const std::vector<std::vector<cv::Point3f>> &obj_points,
                       const Dictionary &dictionary, const std::vector<int> &ids)
{
  return cv::aruco::Board::create(obj_points, dictionary, ids);
}

inline DetectorParameters makeDetectorParameters()
{
  return cv::aruco::DetectorParameters::create();
}

inline void setCornerRefineSubpix(DetectorParameters &params)
{
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  params->doCornerRefinement = true;
#else
  params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif
}

inline void detectMarkers(cv::InputArray image, const Dictionary &dictionary,
                          std::vector<std::vector<cv::Point2f>> &corners,
                          std::vector<int> &ids, const DetectorParameters &params)
{
  cv::aruco::detectMarkers(image, dictionary, corners, ids, params);
}

inline void estimatePoseSingleMarkers(const std::vector<std::vector<cv::Point2f>> &corners,
                                      float marker_length, cv::InputArray camera_matrix,
                                      cv::InputArray dist_coeffs,
                                      std::vector<cv::Vec3d> &rvecs,
                                      std::vector<cv::Vec3d> &tvecs)
{
  cv::aruco::estimatePoseSingleMarkers(corners, marker_length, camera_matrix,
                                       dist_coeffs, rvecs, tvecs);
}

inline int estimatePoseBoard(const std::vector<std::vector<cv::Point2f>> &corners,
                             const std::vector<int> &ids, const Board &board,
                             cv::InputArray camera_matrix, cv::InputArray dist_coeffs,
                             cv::Vec3d &rvec, cv::Vec3d &tvec,
                             bool use_extrinsic_guess = false)
{
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  (void)use_extrinsic_guess;
  return cv::aruco::estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs,
                                      rvec, tvec);
#else
  return cv::aruco::estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs,
                                      rvec, tvec, use_extrinsic_guess);
#endif
}

#endif

// drawDetectedMarkers kept its signature across both APIs; drawAxis was renamed
// to the core cv::drawFrameAxes.
inline void drawDetectedMarkers(cv::InputOutputArray image,
                                const std::vector<std::vector<cv::Point2f>> &corners,
                                const std::vector<int> &ids)
{
  cv::aruco::drawDetectedMarkers(image, corners, ids);
}

inline void drawAxis(cv::InputOutputArray image, cv::InputArray camera_matrix,
                     cv::InputArray dist_coeffs, cv::InputArray rvec,
                     cv::InputArray tvec, float length)
{
  cv::drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length);
}

}  // namespace fc_aruco

#endif  // ARUCO_COMPAT_H
