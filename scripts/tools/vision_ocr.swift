// Text detection with Apple's Vision framework (on-device). Usage: vision_ocr <image> ...
// Prints one JSON line per image: {"file": ..., "words": [{"text", "conf", "box": [x, y, w, h] (normalised, origin top-left)}]}
import Foundation
import Vision
import AppKit

for path in CommandLine.arguments.dropFirst() {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("{\"file\": \"\(path)\", \"error\": \"unreadable\"}"); continue
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.minimumTextHeight = 0.01
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
    var words: [[String: Any]] = []
    for obs in (req.results ?? []) {
        guard let c = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        words.append(["text": c.string, "conf": c.confidence, "box": [b.minX, 1 - b.maxY, b.width, b.height]])
    }
    let data = try! JSONSerialization.data(withJSONObject: ["file": path, "words": words])
    print(String(data: data, encoding: .utf8)!)
}
