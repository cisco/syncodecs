/******************************************************************************
 * Copyright 2014-2016 cisco Systems, Inc.                                    *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *     http://www.apache.org/licenses/LICENSE-2.0                             *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

/**
 * @file
 * Syncodecs implementation file.
 *
 * @version 0.1.0
 * @author Sergio Mena
 * @author Stefano D'Aronco
 * @author Xiaoqing Zhu
 */

#include "syncodecs.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <sys/stat.h>

#define INITIAL_RATE 100.  // Initial (very low) target rate set by default in codecs
#define EPSILON 1e-10  // Used to check floats/doubles for zero
#define RAND_UNIFORM_MAX_RATIO .10  // Width of random variation (noise) in frame size (=10%)

namespace syncodecs {

Codec::Codec() :
    m_targetRate(INITIAL_RATE), m_currentPacketOrFrame(std::vector<uint8_t>(0, 0), 0.) {}

Codec::~Codec() {}

const Codec::value_type Codec::operator*() const {
    assert(isValid());
    return m_currentPacketOrFrame;
}

const Codec::value_type* Codec::operator->() const {
    assert(isValid());
    return &m_currentPacketOrFrame;
}

Codec& Codec::operator++() {
    assert(isValid());
    nextPacketOrFrame(); //Update current packet/frame
    return *this;
}

Codec::operator bool() const {
    return isValid();
}

float Codec::getTargetRate() const {
    return m_targetRate;
}

float Codec::setTargetRate(float newRateBps) {
    if (newRateBps > EPSILON) {
        m_targetRate = newRateBps;
    }
    return m_targetRate;
}

bool Codec::isValid() const {
    return m_currentPacketOrFrame.first.size() > 0;
}



CodecWithFps::CodecWithFps(double fps) : Codec(), m_fps(fps) {
    assert(fps > 0);
}

CodecWithFps::~CodecWithFps() {}



const float CodecWithFpsAndRandomness::m_randMaxRatio = RAND_UNIFORM_MAX_RATIO;

CodecWithFpsAndRandomness::CodecWithFpsAndRandomness(double fps) : CodecWithFps(fps) {
}

CodecWithFpsAndRandomness::~CodecWithFpsAndRandomness() {}

float CodecWithFpsAndRandomness::addNoiseDefault(float size) {
    size += size * uniform(-m_randMaxRatio, m_randMaxRatio);
    return size;
}

double CodecWithFpsAndRandomness::uniform(double min, double max) {
    return ((double)rand() / (double)RAND_MAX) * (max - min) + min;
}



Packetizer::Packetizer(unsigned long payloadSize) :
    Codec(), m_payloadSize(payloadSize) {
    assert(payloadSize > 0);
}

Packetizer::~Packetizer() {}






PerfectCodec::PerfectCodec(unsigned long payloadSize) :
    Packetizer(payloadSize) {
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

PerfectCodec::~PerfectCodec() {}

void PerfectCodec::nextPacketOrFrame() {
    const double secsToNextFrame = static_cast<double>(m_payloadSize) * 8. / m_targetRate;

    m_currentPacketOrFrame.first.resize(m_payloadSize, 0);
    m_currentPacketOrFrame.second = secsToNextFrame;
}



SimpleFpsBasedCodec::SimpleFpsBasedCodec(double fps) :
    CodecWithFps(fps) {
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

SimpleFpsBasedCodec::~SimpleFpsBasedCodec() {}

void SimpleFpsBasedCodec::nextPacketOrFrame() {
    const unsigned long frameBytes = std::ceil(m_targetRate / (m_fps * 8.));
    assert(frameBytes > 0);

    const double secsToNextFrame = 1. / m_fps;

    m_currentPacketOrFrame.first.resize(frameBytes, 0);
    m_currentPacketOrFrame.second = secsToNextFrame;
}



TraceBasedCodec::Labels2Res TraceBasedCodec::m_labels2Res;
const float TraceBasedCodec::m_lowBppThresh = .091;
const float TraceBasedCodec::m_highBppThresh = .175;


TraceBasedCodec::TraceBasedCodec(const std::string& path,
                                 const std::string& filePrefix,
                                 double fps,
                                 bool fixed) :
    CodecWithFps(fps), m_fixedModeEnabled(fixed), m_currentFrameIdx(0) {
    static class FillResolutions_ {
        void addLabelAndResolution(const ResLabel& label, const Resolution reso) {
            m_labels2Res.push_back(std::make_pair(label, reso));
        }
    public:
        FillResolutions_() {
            assert(m_labels2Res.empty());
            addLabelAndResolution("90p", std::make_pair(160, 90));
            addLabelAndResolution("180p", std::make_pair(320, 180));
            addLabelAndResolution("240p", std::make_pair(352, 240));
            addLabelAndResolution("360p", std::make_pair(640, 360));
            addLabelAndResolution("480p", std::make_pair(640, 480));
            addLabelAndResolution("540p", std::make_pair(960, 540));
            addLabelAndResolution("720p", std::make_pair(1280, 720));
            addLabelAndResolution("1080p", std::make_pair(1920, 1080));
        }
    } fillRes_;
    readTraceDataFromDir(path, filePrefix);
    setResolutionForFixedMode();
    if (m_fixedModeEnabled) {
        setFixedMode(true); // Start with the middle resolution
    }
    assert(traceDataIsValid());
    m_limitPixelsPerFrame = getPixelsPerFrame("480p");
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

TraceBasedCodec::~TraceBasedCodec() {}


void TraceBasedCodec::setFixedMode(bool fixed) {
    m_fixedModeEnabled = fixed;
    if (fixed) {
        assert(m_traceData.find(*m_fixedResIt) != m_traceData.end());
        m_currentResIt = m_fixedResIt;
    }
}

bool TraceBasedCodec::getFixedMode() const {
    return m_fixedModeEnabled;
}

void TraceBasedCodec::setResolutionForFixedMode() {
    std::vector<ResLabel>::const_iterator it = m_resolutions.begin();
    std::advance(it, m_resolutions.size() / 2);
    assert(it != m_resolutions.end());
    assert(m_traceData.find(*it) != m_traceData.end());
    m_fixedResIt = it;
}

bool TraceBasedCodec::setResolutionForFixedMode(ResLabel resolution) {
    std::vector<ResLabel>::const_iterator it = std::find(m_resolutions.begin(),
                                                         m_resolutions.end(),
                                                         resolution);
    if (it == m_resolutions.end()) {
        return false;
    }
    assert(m_traceData.find(resolution) != m_traceData.end());
    m_fixedResIt = it;
    return true;
}

bool TraceBasedCodec::isValid() const {
    return traceDataIsValid() && CodecWithFps::isValid();
}

void TraceBasedCodec::decreaseResolution() {
    //PrintResolutionAndBitrate ();
    if (m_currentResIt != m_resolutions.begin()) {
        --m_currentResIt;
        matchBitrate();
        printResolutionAndBitrate();
        std::cout << "Resolution decreased to " << *m_currentResIt
                  << " (new bpp: " << getCurrentBpp() << ")" << std::endl;
    } else {
        //std::cout << "Resolution not decreased: already at minimum" << std::endl;
    }
}

void TraceBasedCodec::increaseResolution() {
    //PrintResolutionAndBitrate ();
    ++m_currentResIt;
    if (m_currentResIt != m_resolutions.end()) {
        matchBitrate();
        const double newBpp = getCurrentBpp();
        if (newBpp < m_lowBppThresh) {
            //std::cout << "Resolution not increased to " << *m_currentResIt
            //          << ", new bpp would be too low: " << newBpp
            //          << " < " << m_lowBppThresh << std::endl;
            --m_currentResIt;
            matchBitrate();
        } else {
            printResolutionAndBitrate();
            std::cout << "Resolution increased to " << *m_currentResIt
                      << " (new bpp: " << newBpp << ")" << std::endl;
        }
    } else {
        --m_currentResIt;
        //std::cout << "Resolution not increased: already at maximum" << std::endl;
    }
}

void TraceBasedCodec::adjustResolution() {
    matchBitrate();
    if (!m_fixedModeEnabled) {
        const double bpp = getCurrentBpp();
        if (bpp < m_lowBppThresh) {
            //Image quality is poor
            //std::cout << "Current bpp too low (" << bpp << " < " << m_lowBppThresh << "). ";
            decreaseResolution();
        } else if (bpp > m_highBppThresh) {
            //Image won't improve with extra bitrate
            //std::cout << "Current bpp too high (" << bpp << " > " << m_highBppThresh << "). ";
            increaseResolution();
        }
    }
}

unsigned long TraceBasedCodec::getFrameBytes(Bitrate rate) {
    const FrameSequence& seq = m_traceData.at(*m_currentResIt)[rate];
    assert(seq.size() > N_FRAMES_EXCLUDED);
    if (m_currentFrameIdx >= seq.size()) {
        m_currentFrameIdx = N_FRAMES_EXCLUDED;
    }
    const unsigned long frameBytes = seq[m_currentFrameIdx++].m_size;
    assert(frameBytes > 0);
    return frameBytes;
}


void TraceBasedCodec::nextPacketOrFrame() {
    adjustResolution();

    m_currentPacketOrFrame.first.resize(getFrameBytes(m_matchedRate), 0);

    const double secsToNextFrame = 1. / m_fps;
    m_currentPacketOrFrame.second = secsToNextFrame;
}

bool TraceBasedCodec::traceDataIsValid() const {
    bool result = !m_resolutions.empty() && m_currentResIt != m_resolutions.end();
    assert(m_traceData.find(*m_currentResIt) != m_traceData.end());
    return result;
}

double TraceBasedCodec::getPixelsPerFrame(ResLabel resolution) {
    Labels2Res::const_iterator it;
    for (it = m_labels2Res.begin(); it->first != resolution && it != m_labels2Res.end(); ++it) {
    }
    assert(it != m_labels2Res.end());
    return it->second.first * it->second.second;
}

void TraceBasedCodec::getBppData(double& scalingFactor, double& targetPixelsPerFrame) const {
    double pixelsPerFrame = getPixelsPerFrame(*m_currentResIt);

    // Above 480p, we apply Waggoner's ^.75 rule
    if (pixelsPerFrame > m_limitPixelsPerFrame) {
        scalingFactor = pow((pixelsPerFrame / m_limitPixelsPerFrame), .75);
        targetPixelsPerFrame = m_limitPixelsPerFrame;
    } else {
        scalingFactor = 1.;
        targetPixelsPerFrame = pixelsPerFrame;
    }
}

double TraceBasedCodec::getCurrentBpp() const {
    double scalingFactor;
    double targetPixelsPerFrame;
    getBppData(scalingFactor, targetPixelsPerFrame);

    return static_cast<double>(m_matchedRate) / (targetPixelsPerFrame * m_fps) * scalingFactor;
}

void TraceBasedCodec::printResolutionAndBitrate() const {
    std::cout << "Resolution is " << *m_currentResIt << ", bitrate is " << m_matchedRate << ". ";
}

void TraceBasedCodec::matchBitrate() {
    // Look up appropriate bitrate
    BitrateMap::const_reverse_iterator it;
    const BitrateMap& currentMap = m_traceData.at(*m_currentResIt);
    // Find greatest rate less than the target rate
    // Both stored and target bitrates are in bps
    for (it = currentMap.rbegin();
         it != currentMap.rend() && static_cast<float>(it->first) > m_targetRate;
         ++it) {
    }
    m_matchedRate = (it != currentMap.rend() ? it->first : currentMap.begin()->first);
}

void TraceBasedCodec::readTraceDataFromDir(const std::string& path, const std::string& filePrefix) {
    for (Labels2Res::const_iterator it = m_labels2Res.begin(); it != m_labels2Res.end(); ++it) {
        bool resolutionPresent = false;
        for (Bitrate bitrate = TRACE_MIN_BITRATE;
             bitrate < TRACE_MAX_BITRATE;
             bitrate += TRACE_BITRATE_STEP) {
            std::ostringstream fullName;
            fullName << path << "/" << filePrefix << "_" << it->first << "_" << bitrate << ".txt";
            struct stat buffer;
            if (::stat(fullName.str().c_str(), &buffer) == 0) { //filename exists
                //* 1000: from kbps to bps
                readTraceDataFromFile(fullName.str(), it->first, bitrate * 1000);
                resolutionPresent = true;
            }
        }
        if (resolutionPresent) {
            m_resolutions.push_back(it->first);
        }
    }
    // Initialize 1st layer index to lowest resolution found
    assert(!m_resolutions.empty()); //TODO: Turn this into meaningful error
    m_currentResIt = m_resolutions.begin();
}

void TraceBasedCodec::readTraceDataFromFile(const std::string& filename, const ResLabel& resolution, Bitrate bitrate) {
    std::ifstream fin(filename.c_str());
    assert(fin);

    std::cout << "Reading traces file " << filename << std::endl;
    FrameDataIterator it(fin);

    while (it) {
        const FrameDataIterator::value_type r = *it;
        m_traceData[resolution][bitrate].push_back(r);
        ++it;
    }
}



TraceBasedCodecWithScaling::TraceBasedCodecWithScaling(const std::string& path,
                                                       const std::string& filePrefix,
                                                       double fps,
                                                       bool fixed) :
    TraceBasedCodec(path, filePrefix, fps, fixed), m_lowRate(0), m_highRate(0) {}

TraceBasedCodecWithScaling::~TraceBasedCodecWithScaling() {}

void TraceBasedCodecWithScaling::nextPacketOrFrame() {
    adjustResolution();

    unsigned long frameBytes;
    assert(m_lowRate != 0 || m_highRate != 0); // At least one rate must be valid
    if (m_lowRate == 0 ) {
        assert(m_targetRate < m_highRate);
        //Linear scaling
        const double scalingFactor = m_targetRate / static_cast<double>(m_highRate);
        // We set the minimum to 1 byte, since scalingFactor can be arbitrarily close to 0
        frameBytes = std::max(1UL, (unsigned long)(scalingFactor * (double)getFrameBytes(m_highRate)));
    } else if (m_highRate == 0) {
        assert(m_lowRate <= m_targetRate);
        //Linear scaling
        const double scalingFactor = m_targetRate / static_cast<double>(m_lowRate);
        frameBytes = scalingFactor * getFrameBytes(m_lowRate);
    } else { // both low rate and high rate are valid
        assert(m_lowRate <= m_targetRate);
        assert(m_targetRate < m_highRate);

        const FrameSequence& lowSeq = m_traceData.at(*m_currentResIt)[m_lowRate];
        assert(lowSeq.size() > N_FRAMES_EXCLUDED);
        if (m_currentFrameIdx >= lowSeq.size()) {
            m_currentFrameIdx = N_FRAMES_EXCLUDED;
        }

        const FrameSequence& highSeq = m_traceData.at(*m_currentResIt)[m_highRate];
        // Frame sequence should be the same, otherwise it doesn't make sense to interpolate
        assert(lowSeq.size() == highSeq.size());

        const double lowSize = lowSeq[m_currentFrameIdx].m_size;
        assert(0 < lowSize);
        const double highSize = highSeq[m_currentFrameIdx++].m_size;
        if (lowSize > highSize) {
            std::cout << "Warning: Frame size (" << lowSize << ")@" << m_lowRate <<
            " is bigger than size (" << highSize << ")@" << m_highRate << std::endl;
        }

        const double highLowDiff = m_highRate - m_lowRate;
        const double targetLowDiff = m_targetRate - m_lowRate;
        const double highWeight = targetLowDiff / highLowDiff;
        //Linear interpolation
        frameBytes = highSize * highWeight + lowSize * (1. - highWeight);
    }

    const double secsToNextFrame = 1. / m_fps;

    m_currentPacketOrFrame.first.resize(frameBytes, 0);
    m_currentPacketOrFrame.second = secsToNextFrame;
}

double TraceBasedCodecWithScaling::getCurrentBpp() const {
    double scalingFactor;
    double targetPixelsPerFrame;
    getBppData(scalingFactor, targetPixelsPerFrame);

    return static_cast<double>(m_targetRate) / (targetPixelsPerFrame * m_fps) * scalingFactor;
}

void TraceBasedCodecWithScaling::printResolutionAndBitrate() const {
    std::cout << "Resolution is " << *m_currentResIt << ", bitrate is " << m_targetRate << ". ";
}

void TraceBasedCodecWithScaling::matchBitrate() {
    // Find rates immediately lower and immediately higher than the target rate
    // m_lowRate <- 0 if it cannot be lower than target rate
    // m_highRate <- 0 if it cannot be greater than target rate
    // A rate set to 0 means "invalid"
    // All bitrates are in bps
    const BitrateMap& currentMap = m_traceData.at(*m_currentResIt);
    assert(currentMap.size() > 0);
    m_lowRate = 0;
    BitrateMap::const_iterator it;
    for (it = currentMap.begin();
         it != currentMap.end() && static_cast<float>(it->first) <= m_targetRate;
         ++it) {
        m_lowRate = it->first;
    }
    m_highRate = (it != currentMap.end() ? it->first : 0);
}


ShapedPacketizer::ShapedPacketizer(Codec* innerCodec,
                                   unsigned long payloadSize,
                                   unsigned int perPacketOverhead) :
    Packetizer(payloadSize),
    m_innerCodec(innerCodec),
    m_overhead(perPacketOverhead),
    m_bytesToSend(0, 0),
    m_secsToNextFrame(0.),
    m_lastOverheadFactor((double)perPacketOverhead / (double)payloadSize) {

    assert(innerCodec != NULL);
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

ShapedPacketizer::~ShapedPacketizer() {}

bool ShapedPacketizer::isValid() const {
    return Codec::isValid() && (bool)m_innerCodec.get();
}

void ShapedPacketizer::nextPacketOrFrame() {
    if (m_bytesToSend.size() == 0) {
        assert(std::abs(m_secsToNextFrame) < EPSILON);

        Codec &codec = *m_innerCodec;
        m_innerCodec->setTargetRate(m_targetRate / (1. + m_lastOverheadFactor));
        ++codec; // Advance codec to next frame
        m_bytesToSend = codec->first;
        m_secsToNextFrame += codec->second;

        const double packetsToSend =
            std::ceil((double)m_bytesToSend.size() / (double)(m_payloadSize));
        assert(m_bytesToSend.size() > 0);
        m_lastOverheadFactor = (double)m_overhead * packetsToSend / (double)m_bytesToSend.size();
    }

    assert(m_bytesToSend.size() > 0);
    assert(m_secsToNextFrame >= 0);

    // m_payloadSize is interpreted here as "max payload size"
    const unsigned long payloadSize = std::min(m_payloadSize, m_bytesToSend.size());
    const double packetsToSend = std::ceil((double)m_bytesToSend.size() / (double)(m_payloadSize));
    assert(packetsToSend >= 1.);
    const double secsToNextPacket = m_secsToNextFrame / packetsToSend;

    // copy the first part of the vector
    m_currentPacketOrFrame.first =
        std::vector<uint8_t>(m_bytesToSend.begin(), m_bytesToSend.begin() + payloadSize);
    m_currentPacketOrFrame.second = secsToNextPacket;

    // remove the first part of the vector
    m_bytesToSend = std::vector<uint8_t>(m_bytesToSend.begin() + payloadSize, m_bytesToSend.end());
    m_secsToNextFrame -= secsToNextPacket;
}






StatisticsCodec::StatisticsCodec(double fps,
                                 AddNoiseFunc addNoise,
                                 float maxUpdateRatio,
                                 double updateInterval,
                                 float bigChangeRatio,
                                 unsigned int transientLength,
                                 float iFrameRatio) :
        CodecWithFpsAndRandomness(fps), m_maxUpdateRatio(maxUpdateRatio),
        m_updateInterval(updateInterval),
        m_bigChangeRatio(bigChangeRatio), m_transientLength(transientLength),
        m_iFrameRatio(iFrameRatio), m_timeToUpdate(0.),
        m_remainingBurstFrames (transientLength), // Start with a burst
        m_addNoise(addNoise) {
    assert(m_maxUpdateRatio > -EPSILON); // >= 0
    assert(m_updateInterval > -EPSILON); // >= 0
    assert(m_bigChangeRatio > EPSILON); // > 0
    assert(m_iFrameRatio > EPSILON); // > 0
    assert(m_addNoise != NULL);
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

StatisticsCodec::~StatisticsCodec() {}

float StatisticsCodec::setTargetRate(float newRateBps) {
    if (newRateBps < EPSILON || m_timeToUpdate > EPSILON) {
        return m_targetRate;
    }

    // Will have to wait the update interval before accepting a new update
    m_timeToUpdate = m_updateInterval;

    // Big change, initiate burst
    const float changeFact = (newRateBps - m_targetRate) / m_targetRate;
    if (std::abs(changeFact) > m_bigChangeRatio) {
        m_remainingBurstFrames = m_transientLength;
        m_targetRate = newRateBps;
        return m_targetRate;
    }

    // Not big change, so clip to +/- m_maxUpdateRatio
    if (m_maxUpdateRatio > EPSILON) {
        const double upperBound = m_targetRate * (1 + m_maxUpdateRatio);
        const double lowerBound = m_targetRate * (1 - m_maxUpdateRatio);
        if (newRateBps > upperBound) {
            newRateBps = upperBound;
        } else if (newRateBps < lowerBound) {
            newRateBps = std::max(INITIAL_RATE, lowerBound);
        }
    }
    m_targetRate = newRateBps;

    return m_targetRate;
}

void StatisticsCodec::nextPacketOrFrame() {
    float frameBytes = m_targetRate / (m_fps * 8.);
    if (m_remainingBurstFrames > 0) {
        assert(m_transientLength > 0);
        if (m_remainingBurstFrames == m_transientLength) { // I frame
            frameBytes *= m_iFrameRatio;
        } else {
            float newRatio = ((float)m_transientLength) - m_iFrameRatio;
            newRatio /= ((float)(m_transientLength - 1));
            newRatio = std::max(.2f, newRatio);
            frameBytes *= newRatio;
        }
        --m_remainingBurstFrames;
    }

    // Apply the configured function for noise
    frameBytes = m_addNoise(frameBytes);
    // Should have at least 1 byte to send
    frameBytes = std::max(1.f, frameBytes);

    const double secsToNextFrame = 1. / m_fps;

    m_currentPacketOrFrame.first.resize((size_t)frameBytes, 0);
    m_currentPacketOrFrame.second = secsToNextFrame;

    m_timeToUpdate = std::max(0., m_timeToUpdate - secsToNextFrame);
}





SimpleContentSharingCodec::SimpleContentSharingCodec(double fps,
                                 unsigned long noChangeMaxSize,
                                 float bigFrameProb,
                                 float bigFrameRatioMin,
                                 float bigFrameRatioMax) :
        CodecWithFpsAndRandomness(fps),
        m_noChangeMaxSize(noChangeMaxSize),
        m_bigFrameProb(bigFrameProb),
        m_bigFrameRatioMin(bigFrameRatioMin),
        m_bigFrameRatioMax(bigFrameRatioMax),
        m_first(true) {
    assert(m_noChangeMaxSize > 0);
    assert(m_bigFrameProb > -EPSILON); // >= 0%
    assert(m_bigFrameProb <= 1.); // <= 100%
    assert(m_bigFrameRatioMin > -EPSILON); // >= 0
    assert(m_bigFrameRatioMax > -EPSILON); // >= 0
    nextPacketOrFrame(); //Read first frame
    assert(isValid());
}

SimpleContentSharingCodec::~SimpleContentSharingCodec() {}

void SimpleContentSharingCodec::nextPacketOrFrame() {
    float frameBytes = m_targetRate / (m_fps * 8.f);

    // Cap bytes to maximum small packet size
    frameBytes = std::min(frameBytes, (float)m_noChangeMaxSize);

    // Time for a big frame?
    if (m_first || uniform(0., 1.) < m_bigFrameProb) {
        m_first = false;
        frameBytes *= uniform(m_bigFrameRatioMin, m_bigFrameRatioMax);
    }

    // Should have at least 1 byte to send
    frameBytes = std::max(1.f, frameBytes);

    const double secsToNextFrame = 1. / m_fps;

    m_currentPacketOrFrame.first.resize((size_t)frameBytes, 0);
    m_currentPacketOrFrame.second = secsToNextFrame;
}


}
