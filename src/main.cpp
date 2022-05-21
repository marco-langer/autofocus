/**
 * Example on how to extract frames from a video:
 *
 *    ffmpeg -i <input_file> <frame_name%05d.png> -hide_banner
 *
 * ATTENTION: this script expects the frame number to have 5 digits.
 */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fmt/core.h>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Result of frame analysis.
 *
 * The sharpness is defined as a real value: a low value means less sharpness
 * and a high value means higher sharpness.
 */
struct frame_info
{
    std::size_t number{}; /* frame number */
    double sharpness{};   /* frame sharpness */
};

/**
 * Extracts the frame number from the given filepath and returns it.
 *
 * The image file paths are expected to have following signatur:
 *    'filename%05d.<extension>'
 */
auto extract_frame_number(
    std::filesystem::path const & filepath,
    std::size_t frame_number_digits = 5) -> std::size_t
{
    auto const extensionSize = filepath.extension().u8string().size();
    auto const str = filepath.u8string();
    if (str.length() <= extensionSize + frame_number_digits)
    {
        throw std::runtime_error{
            "invalid filename: '{}'",
            reinterpret_cast<char const *>(str.c_str())
        };
    }
    auto const start_index = str.length() - extensionSize - frame_number_digits;
    auto frame_number = std::size_t{};
    auto [ptr, ec] = std::from_chars(
        reinterpret_cast<char const *>(str.c_str()) + start_index,
        reinterpret_cast<char const *>(str.c_str()) + str.length(),
        frame_number
    );
    if (ec != std::errc{})
    {
        throw std::runtime_error{
            fmt::format(
                "unable to parse frame number from file '{}'",
                reinterpret_cast<char const *>(str.c_str())
            )
        };
    }

    return frame_number;
}

/**
 * Reads an image from the given file path and returns it as a cv::Mat object.
 */
auto read_image(std::filesystem::path const & filepath) -> cv::Mat
{
    auto const image = cv::imread(
        reinterpret_cast<const char *>(filepath.u8string().c_str())
    );
    if (image.cols == 0 || image.rows == 0)
    {
        throw std::runtime_error{
            fmt::format(
                "error while opening image '{}'",
                reinterpret_cast<char const *>(filepath.u8string().c_str())
            )
        };
    }
    return image;
}

/**
 * Reads the image at the given file path and returns the 'sharpness' as a real number.
 */
auto calculate_sharpness(std::filesystem::path const & filepath) -> double
{
    /**
     * edge detection: algorithm taken from
     * https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
     */
    auto image = read_image(filepath);

    // apply gaussian blur to suppress noise and convert to gray
    cv::GaussianBlur(image, image, cv::Size{3, 3}, 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // use laplacian folding for edge detection
    auto filtered_image = cv::Mat{};
    cv::Laplacian(image, filtered_image, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);

    // find maximum pixel value in the filtered image. This is our sharpeness value
    auto min_val = double{};
    auto max_val = double{};
    auto min_loc = cv::Point{};
    auto max_loc = cv::Point{};
    cv::minMaxLoc(filtered_image, &min_val, &max_val, &min_loc, &max_loc);

    return max_val;
}

/**
 * Analysis the image from the given path and returns the result as a frame_info.
 */
auto analyse_frame(std::filesystem::path const & filepath) -> frame_info
{
    return frame_info{
        .number = extract_frame_number(filepath),
        .sharpness = calculate_sharpness(filepath)
    };
}

/**
 * Analysis all images in the given directory and returns the result as a vector of frame_info.
 */
auto analyse_frames(std::filesystem::path const & directory) -> std::vector<frame_info>
{
    if (!std::filesystem::is_directory(directory))
    {
        throw std::runtime_error{
            fmt::format(
                "invalid data directory '{}'.",
                reinterpret_cast<char const *>(directory.u8string().c_str())
            )
        };
    }

    std::vector<frame_info> result;
    for (auto && entry : std::filesystem::directory_iterator{directory})
    {
        result.emplace_back(analyse_frame(entry.path()));
    }

    // directory_iterator does not iterate sorted
    std::sort(std::begin(result), std::end(result),
        [](auto const & lhs_frame, auto const & rhs_frame)
        {
            return lhs_frame.number < rhs_frame.number;
        }
    );

    return result;
}

/**
 * Writes the frame infos to the given file path as a tab-delimited ASCII table.
 */
void write_results(
    std::filesystem::path const & filepath,
    std::vector<frame_info> frames)
{
    auto ofs = std::ofstream(filepath.string());
    if (!ofs)
    {
        throw std::runtime_error{
            fmt::format(
                "unable to open result file '{}'",
                reinterpret_cast<char const *>(filepath.u8string().c_str())
            )
        };
    }

    for (auto && frame : frames)
    {
        ofs << fmt::format("{}\t{}\n", frame.number, frame.sharpness);
    }
}

struct parsed_args
{
    std::filesystem::path frames_directory;
    std::filesystem::path result_file;
};

auto parse_args(int argc, char ** argv) -> parsed_args
{
    if (argc != 3)
    {
        throw std::runtime_error{
            "invalid arguments. Usage:\nautofocus <frames_directory> <result_filepath"
        };
    }

    return parsed_args{
        .frames_directory = std::filesystem::u8path(argv[1]),
        .result_file = std::filesystem::u8path(argv[2])
    };
}

auto main(int argc, char ** argv) -> int
{
    try
    {
        auto const [frames_directory, result_file] = parse_args(argc, argv);
        write_results(result_file, analyse_frames(frames_directory));
    }
    catch (std::exception const & e)
    {
        std::cout << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}