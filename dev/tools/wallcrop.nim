# wallcrop.nim - crop a huge image into wallpapers that are pixel-perfect or pixel-group-perfect.
#
# What it does
#   You choose an output resolution (a named screen size, or custom) and a scale factor n.
#   The tool shows a rectangle (outline) of n*W x n*H pixels of the source image, which you move
#   with the keys.  Exporting turns every n x n block of the rectangle into ONE output pixel, by
#   one of eleven sampling modes (the mean, a channelwise median, max, min, a random pixel, the top
#   left pixel, the pixel farthest from the mean ..., or all of them: one image each) - integer
#   arithmetic, no filtering:
#     n = 1  -> the pixels are copied (pixel-perfect: no resampling at all)
#     n > 1  -> pixel-group-perfect: exactly n x n source pixels per output pixel, nothing else.
#
# Run it (this file is both the source and the program; nim compiles and runs it):
#     nim r -d:release dev/tools/wallcrop.nim [image]
#   other modes:  --selftest | --export ... | --snapshot ... | --help
#
# Needs
#   * Nim 2.x with a C compiler (clang or gcc);
#   * libvips (the vips and vipsheader programs): decodes any image once into a cache of raw pixels
#     and writes the exported PNG.  Set VipsDir below, or put the programs on PATH;
#   * SDL2 (64-bit) to open the window: SDL2.dll on Windows (searched for next to the program,
#     next to this file, in the wallcrop folder of the user's application data, then on PATH),
#     libSDL2 on Linux (the package manager's libsdl2-2.0-0 / SDL2).
#
# How big images are handled
#   The first time an image is opened, vips converts it to a raw file of 8-bit pixels (3 channels,
#   or 4 when the source has alpha) in the cache folder; the program then memory-maps it, so only
#   the pixels on screen are ever read, and a half-size pyramid is built in the background for the
#   zoomed-out views.  Opening the same file again is instant.  The cache is never deleted by the
#   program (it can be several GB per image): the folder is shown with --help.
#
# Sources that are not 8-bit sRGB (16-bit, grey, CMYK ...) are converted to 8-bit sRGB by vips first;
# 8-bit RGB and RGBA sources are used exactly as they are.  Averaging is done on the stored
# (gamma-encoded) values, as nearly every image tool does.
#
# The block "remembered by the program" below is REWRITTEN BY THE PROGRAM ITSELF: whenever you open
# an image or pick a save folder it edits those two lines of this very file.  (So every later
# `nim r` of the file compiles again, and the lines show up in version control diffs.)

import std/[os, strutils, osproc, streams, memfiles, monotimes, times, atomics, math, algorithm,
            typedthreads, random, dynlib]
when defined(windows):
  from std/unicode import runes, toUTF8, Rune        # only the Windows dialogs need UTF-16 conversion

{.passC: "-O3".}

# ---------------------------------------------------------------------------------------------
# Settings you may edit
# ---------------------------------------------------------------------------------------------
const
  VipsDir = r"E:\data-game-win10pro2\lock\apps\lib\vips-dev-8.18\bin"  ## folder with vips and vipsheader; "" = search PATH
  SdlLibrary = ""         ## full path of the SDL2 library; "" = search (see findSdl)
  CacheDirOverride = ""   ## where the converted images live; "" = <application data>/wallcrop/cache
  MaxScale = 8            ## largest n offered (n x n source pixels per output pixel)
  CameraMargin = 1.25     ## the normal camera shows the rectangle this much smaller than the window
  HoldStartSpeed = 120.0  ## source pixels per second when a held arrow key starts to speed up (it then doubles every 0.35 s)

# ---------------------------------------------------------------------------------------------
# Remembered by the program (it rewrites these two lines of this file itself - do not reformat)
# ---------------------------------------------------------------------------------------------
const LastImage = ""
const LastSaveDir = ""

const
  SourceFile = currentSourcePath()
  CacheVersion = 1
  MaxLevels = 12

type
  Bytes = ptr UncheckedArray[uint8]
  Pixels = ptr UncheckedArray[uint32]

proc byteAt(p: Bytes, i: int): Bytes {.inline.} = cast[Bytes](cast[uint](p) + uint(i))

# ---------------------------------------------------------------------------------------------
# Output resolutions
# ---------------------------------------------------------------------------------------------
type Res = tuple[w, h: int, name: string]

const Resolutions: array[25, Res] = [
  (1280, 720, "HD 720p"),
  (1366, 768, "HD laptop"),
  (1600, 900, "HD+"),
  (1920, 1080, "Full HD"),
  (1920, 1200, "WUXGA"),
  (2048, 1080, "DCI 2K"),
  (2560, 1080, "UltraWide FHD"),
  (2560, 1440, "QHD"),
  (2560, 1600, "WQXGA"),
  (2880, 1800, "Retina 15 inch"),
  (3200, 1800, "QHD+"),
  (3440, 1440, "UltraWide QHD"),
  (3840, 1080, "Dual Full HD"),
  (3840, 1600, "UltraWide QHD+"),
  (3840, 2160, "4K UHD"),
  (4096, 2160, "DCI 4K"),
  (5120, 1440, "Dual QHD"),
  (5120, 2160, "5K2K UltraWide"),
  (5120, 2880, "5K"),
  (6016, 3384, "6K"),
  (7680, 4320, "8K UHD"),
  (1080, 1920, "Phone Full HD"),
  (1080, 2400, "Phone 20:9"),
  (1170, 2532, "iPhone 12 to 14"),
  (1440, 3200, "Phone QHD+")]

const CustomIndex = Resolutions.len     ## resolution index meaning "custom"

# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
proc clampInt(v, lo, hi: int): int {.inline.} = (if v < lo: lo elif v > hi: hi else: v)

proc fnv64(s: string): uint64 =
  ## FNV-1a: a hash that never changes between runs or Nim versions (names the cache folders).
  result = 14695981039346656037'u64
  for c in s:
    result = (result xor uint64(ord(c))) * 1099511628211'u64

proc safeStem(name: string): string =
  ## A file name stem that is safe in a vips file name ([ ] start options) and on every file system.
  for c in name:
    if c in {'a'..'z', 'A'..'Z', '0'..'9', '-', '_', '.', '(', ')', ' '}: result.add c
    else: result.add '_'
  if result.len == 0: result = "image"
  if result.len > 60: result = result[0 ..< 60]

proc fmtBytes(n: int64): string =
  if n >= (1'i64 shl 30): formatFloat(n.float / float(1 shl 30), ffDecimal, 1) & " GB"
  elif n >= (1'i64 shl 20): formatFloat(n.float / float(1 shl 20), ffDecimal, 1) & " MB"
  else: $n & " bytes"

proc truncateMiddle(s: string, maxChars: int): string =
  ## "very/long/path.png" -> "very/lo...path.png" when it does not fit
  if s.len <= maxChars or maxChars < 8: return s
  let keepTail = (maxChars - 3) div 2 + (maxChars - 3) mod 2
  let keepHead = maxChars - 3 - keepTail
  s[0 ..< keepHead] & "..." & s[s.len - keepTail .. ^1]

proc appDir(): string =
  when defined(windows):
    getEnv("LOCALAPPDATA", getHomeDir() / "AppData" / "Local") / "wallcrop"
  else:
    getEnv("XDG_CACHE_HOME", getHomeDir() / ".cache") / "wallcrop"

var cacheRootOverride = ""      ## the self-test points this at a scratch folder

proc cacheRoot(): string =
  if cacheRootOverride.len > 0: cacheRootOverride
  elif CacheDirOverride.len > 0: CacheDirOverride
  else: appDir() / "cache"

# ---------------------------------------------------------------------------------------------
# Remembering paths by editing this very file
# ---------------------------------------------------------------------------------------------
proc replaceConstLine*(text, name, value: string): tuple[text: string, found: bool] =
  ## In `text` (a Nim source) find the line `const <name> = ...` and give it the literal `value`.
  ## Every other byte - including the CRLF or LF ending of the line - is left as it is.
  let prefix = "const " & name & " = "
  var pos = 0
  var output = ""
  while pos < text.len:
    var eol = text.find('\n', pos)
    let lineEnd = if eol < 0: text.len else: eol          # the text of the line ends here
    var contentEnd = lineEnd
    if contentEnd > pos and text[contentEnd - 1] == '\r': dec contentEnd
    if not result.found and text.continuesWith(prefix, pos):
      output.add prefix & escape(value)
      output.add text[contentEnd ..< (if eol < 0: text.len else: eol + 1)]
      result.found = true
    else:
      output.add text[pos ..< (if eol < 0: text.len else: eol + 1)]
    pos = if eol < 0: text.len else: eol + 1
  result.text = output

proc rememberInSource(name, value: string): string =
  ## Edit the `const <name> = ...` line of this file.  Returns "" or a message about what failed.
  if not fileExists(SourceFile): return "source file not found, so nothing was remembered"
  try:
    let (text, found) = replaceConstLine(readFile(SourceFile), name, value)
    if not found: return "no `const " & name & "` line in the source"
    let tmp = SourceFile & ".tmp"
    writeFile(tmp, text)
    moveFile(tmp, SourceFile)
  except CatchableError as e:
    return "could not edit the source: " & e.msg

# ---------------------------------------------------------------------------------------------
# Parallel loops
# ---------------------------------------------------------------------------------------------
type
  RowFn = proc (ctx: pointer, a, b: int) {.nimcall, gcsafe.}
  ForArg = object
    next: Atomic[int]
    total, blk: int
    ctx: pointer
    fn: RowFn
    cancel: ptr Atomic[bool]

proc forWorker(arg: ptr ForArg) {.thread.} =
  while true:
    if arg.cancel != nil and arg.cancel[].load(moRelaxed): return
    let a = arg.next.fetchAdd(1) * arg.blk
    if a >= arg.total: return
    arg.fn(arg.ctx, a, min(arg.total, a + arg.blk))

proc workerCount(): int = max(1, min(32, countProcessors()))

proc parallelFor(total, blk: int, ctx: pointer, fn: RowFn, cancel: ptr Atomic[bool] = nil) =
  ## Run fn(ctx, a, b) over [0, total) in blocks of `blk`, on all processors.
  if total <= 0: return
  var arg = ForArg(total: total, blk: max(1, blk), ctx: ctx, fn: fn, cancel: cancel)
  arg.next.store(0)
  let n = min(workerCount(), (total + arg.blk - 1) div arg.blk)
  if n <= 1:
    forWorker(addr arg)
    return
  var threads = newSeq[Thread[ptr ForArg]](n)
  for i in 0 ..< n: createThread(threads[i], forWorker, addr arg)
  joinThreads(threads)

# ---------------------------------------------------------------------------------------------
# The sampling itself: eleven ways to make one pixel of an n x n block, and the half-size pyramid
# ---------------------------------------------------------------------------------------------
proc selectKth(a: var array[64, uint8], m, k: int): uint8 =
  ## Quickselect: afterwards a[k] is the value that would sit at k if a[0..m-1] were sorted, the
  ## values before it are <= it and the values after it are >= it.
  var lo = 0
  var hi = m - 1
  while lo < hi:
    let pivot = a[(lo + hi) shr 1]
    var i = lo
    var j = hi
    while i <= j:
      while a[i] < pivot: inc i
      while a[j] > pivot: dec j
      if i <= j:
        swap(a[i], a[j])
        inc i
        dec j
    if k <= j: hi = j
    elif k >= i: lo = i
    else: break
  a[k]

proc medianTwice(a: var array[64, uint8], m: int): int =
  ## Twice the median of a[0..m-1] (so it is exact): the middle value doubled when m is odd, else the
  ## sum of the two middle values.
  let k = m shr 1
  let upper = selectKth(a, m, k)
  if (m and 1) == 1: return 2 * int(upper)
  var lower = 0'u8
  for i in 0 ..< k:
    if a[i] > lower: lower = a[i]
  int(lower) + int(upper)

proc medianOf(a: var array[64, uint8], m: int): int =
  ## The median of a[0..m-1]: the middle value when m is odd, else the mean of the two middle
  ## values rounded up.
  (medianTwice(a, m) + 1) shr 1

type SampleMode = enum
  smMean, smMedian, smMax, smMin, smRandomColor, smRandomChannels, smTopLeft,
  smFarthestMeanColor, smFarthestMeanChannel, smFarthestMedianChannel, smAll

const
  ModeLabels: array[SampleMode, string] = [
    "mean", "channelwise median", "channelwise max", "channelwise min", "colorwise random sample",
    "channelwise random samples", "top left", "colorwise farthest from mean",
    "channelwise farthest from mean", "channelwise farthest from median", "all"]
  ModeFileNames: array[SampleMode, string] = [
    "mean", "median", "max", "min", "random-color", "random-channels", "top-left", "far-mean-color",
    "far-mean-channels", "far-median-channels", "all"]
  ModeNotes: array[SampleMode, string] = [
    "the average of each block",
    "the middle value of each block, per colour channel",
    "the largest value of each block, per colour channel",
    "the smallest value of each block, per colour channel",
    "one random pixel of each block",
    "a random pixel of the block for each colour channel",
    "the top left pixel of each block",
    "the pixel whose colour is farthest from the block's mean colour",
    "per colour channel, the value farthest from that channel's mean",
    "per colour channel, the value farthest from that channel's median",
    "one image for each of the modes above"]
  SingleModes = [smMean, smMedian, smMax, smMin, smRandomColor, smRandomChannels, smTopLeft,
                 smFarthestMeanColor, smFarthestMeanChannel, smFarthestMedianChannel]
  RandomSeed = 0x5EEDC0FFEE1234'u64     ## the random modes are a fixed function of the seed and the block's place

proc parseMode(name: string): tuple[ok: bool, mode: SampleMode] =
  for m in SampleMode:
    if ModeFileNames[m] == name.toLowerAscii:
      return (true, m)

proc mix64(z0: uint64): uint64 {.inline.} =
  ## splitmix64's finalizer
  var z = z0 + 0x9E3779B97F4A7C15'u64
  z = (z xor (z shr 30)) * 0xBF58476D1CE4E5B9'u64
  z = (z xor (z shr 27)) * 0x94D049BB133111EB'u64
  z xor (z shr 31)

proc channelSalt(ch: int): uint64 {.inline.} = uint64(ch + 1) * 0xD6E8FEB86659FD93'u64

proc pickIndex(sx, sy: int, salt: uint64, count: int): int {.inline.} =
  ## Which of the `count` pixels of the block whose top left source pixel is (sx, sy) a random mode takes.
  ## A pure function of the block's place (no state, no order, no thread matters), so a crop always exports
  ## the same image.  salt is 0 for a pixel and channelSalt(ch) for a channel.
  let h = mix64(RandomSeed xor (uint64(sx) * 0x9E3779B97F4A7C15'u64) xor (uint64(sy) * 0xC2B2AE3D27D4EB4F'u64) xor salt)
  int(h mod uint64(count))

type
  ExportCtx = object
    src: Bytes
    srcW, bands, x0, y0, outW, outH, n: int
    mode: SampleMode
    dst: Bytes
    rowsDone: ptr Atomic[int]

proc exportRows(ctx: pointer, a, b: int) {.nimcall, gcsafe.} =
  ## Output rows [a, b): every output pixel is made of an n x n block by the sampling mode.
  {.push boundChecks: off, overflowChecks: off, rangeChecks: off.}
  let c = cast[ptr ExportCtx](ctx)
  let n = c.n
  let bands = c.bands
  let nn = n * n
  let stride = c.srcW * bands
  let colours = min(bands, 3)             # "colour" modes look at red, green and blue; alpha follows the chosen pixel
  var blk: array[4, array[64, uint8]]     # the block, one list of nn values per channel
  for oy in a ..< b:
    for ox in 0 ..< c.outW:
      let sx = c.x0 + ox * n
      let sy = c.y0 + oy * n
      let o = (oy * c.outW + ox) * bands
      if n == 1:
        let s = sy * stride + sx * bands
        for ch in 0 ..< bands: c.dst[o + ch] = c.src[s + ch]
      elif c.mode == smMean:
        for ch in 0 ..< bands:
          var sum = 0
          for dy in 0 ..< n:
            let row = (sy + dy) * stride + sx * bands + ch
            for dx in 0 ..< n: sum += int(c.src[row + dx * bands])
          c.dst[o + ch] = uint8((sum + (nn shr 1)) div nn)
      else:
        for dy in 0 ..< n:
          let row = (sy + dy) * stride + sx * bands
          for dx in 0 ..< n:
            for ch in 0 ..< bands: blk[ch][dy * n + dx] = c.src[row + dx * bands + ch]
        case c.mode
        of smMean, smAll: discard                       # (the mean is done above, and "all" is many exports)
        of smMedian:
          for ch in 0 ..< bands:
            var t = blk[ch]
            c.dst[o + ch] = uint8(medianOf(t, nn))
        of smMax:
          for ch in 0 ..< bands:
            var best = 0'u8
            for k in 0 ..< nn:
              if blk[ch][k] > best: best = blk[ch][k]
            c.dst[o + ch] = best
        of smMin:
          for ch in 0 ..< bands:
            var best = 255'u8
            for k in 0 ..< nn:
              if blk[ch][k] < best: best = blk[ch][k]
            c.dst[o + ch] = best
        of smTopLeft:
          for ch in 0 ..< bands: c.dst[o + ch] = blk[ch][0]
        of smRandomColor:
          let k = pickIndex(sx, sy, 0'u64, nn)
          for ch in 0 ..< bands: c.dst[o + ch] = blk[ch][k]
        of smRandomChannels:
          for ch in 0 ..< bands: c.dst[o + ch] = blk[ch][pickIndex(sx, sy, channelSalt(ch), nn)]
        of smFarthestMeanColor:
          var sums: array[3, int]
          for ch in 0 ..< colours:
            for k in 0 ..< nn: sums[ch] += int(blk[ch][k])
          var bestK = 0
          var bestD = -1
          for k in 0 ..< nn:               # distance from the mean, scaled by nn so that it is an integer
            var d = 0
            for ch in 0 ..< colours:
              let e = int(blk[ch][k]) * nn - sums[ch]
              d += e * e
            if d > bestD:                  # the first of equally far pixels wins
              bestD = d
              bestK = k
          for ch in 0 ..< bands: c.dst[o + ch] = blk[ch][bestK]
        of smFarthestMeanChannel:
          for ch in 0 ..< bands:
            var sum = 0
            for k in 0 ..< nn: sum += int(blk[ch][k])
            var bestK = 0
            var bestD = -1
            for k in 0 ..< nn:
              let d = abs(int(blk[ch][k]) * nn - sum)
              if d > bestD:
                bestD = d
                bestK = k
            c.dst[o + ch] = blk[ch][bestK]
        of smFarthestMedianChannel:
          for ch in 0 ..< bands:
            var t = blk[ch]
            let twice = medianTwice(t, nn)  # distance from the median, doubled so that it is an integer
            var bestK = 0
            var bestD = -1
            for k in 0 ..< nn:
              let d = abs(2 * int(blk[ch][k]) - twice)
              if d > bestD:
                bestD = d
                bestK = k
            c.dst[o + ch] = blk[ch][bestK]
    if c.rowsDone != nil: discard c.rowsDone[].fetchAdd(1)
  {.pop.}

proc exportBlocks*(src: Bytes, srcW, bands, x0, y0, outW, outH, n: int, mode: SampleMode, dst: Bytes,
                   rowsDone: ptr Atomic[int] = nil, cancel: ptr Atomic[bool] = nil) =
  ## Fill dst (outW x outH x bands bytes) from the n*outW x n*outH rectangle of src at (x0, y0).
  ## `mode` is one way of sampling; "all" is many exports and is made by the caller.
  doAssert mode != smAll, "smAll is made of single modes"
  var ctx = ExportCtx(src: src, srcW: srcW, bands: bands, x0: x0, y0: y0, outW: outW, outH: outH,
                      n: n, mode: mode, dst: dst, rowsDone: rowsDone)
  parallelFor(outH, 4, addr ctx, exportRows, cancel)

type LevelCtx = object
  src, dst: Bytes
  sw, sh, dw, dh, bands: int

proc levelRows(ctx: pointer, a, b: int) {.nimcall, gcsafe.} =
  ## Pyramid level rows [a, b): each pixel is the mean of a 2 x 2 block (the last row/column of an
  ## odd-sized level repeats itself).
  {.push boundChecks: off, overflowChecks: off, rangeChecks: off.}
  let c = cast[ptr LevelCtx](ctx)
  let bands = c.bands
  for y in a ..< b:
    let r0 = min(2 * y, c.sh - 1) * c.sw * bands
    let r1 = min(2 * y + 1, c.sh - 1) * c.sw * bands
    for x in 0 ..< c.dw:
      let x0 = min(2 * x, c.sw - 1) * bands
      let x1 = min(2 * x + 1, c.sw - 1) * bands
      let o = (y * c.dw + x) * bands
      for ch in 0 ..< bands:
        c.dst[o + ch] = uint8((int(c.src[r0 + x0 + ch]) + int(c.src[r0 + x1 + ch]) +
                               int(c.src[r1 + x0 + ch]) + int(c.src[r1 + x1 + ch]) + 2) shr 2)
  {.pop.}

proc halfCeil(v: int): int {.inline.} = (v + 1) shr 1

proc levelSize(w, h, level: int): (int, int) =
  var a = w
  var b = h
  for _ in 0 ..< level:
    a = halfCeil(a)
    b = halfCeil(b)
  (a, b)

proc levelCountFor(w, h: int): int =
  ## Levels in the pyramid, level 0 (full size) included: halve until the long side is <= 512.
  var a = w
  var b = h
  result = 1
  while max(a, b) > 512 and result < MaxLevels:
    a = halfCeil(a)
    b = halfCeil(b)
    inc result

# ---------------------------------------------------------------------------------------------
# Camera and outline geometry
# ---------------------------------------------------------------------------------------------
type Cam = object
  s: float               ## screen pixels per source pixel
  left, top: int         ## where the rectangle's top-left source pixel lands on the screen
  rwS, rhS: int          ## the rectangle's size on the screen, in pixels

proc makeCam(fbW, fbH, rw, rh: int, exact: bool): Cam =
  ## The rectangle is centred.  `exact` fits it to the window (it fills the window's long side);
  ## otherwise the camera is CameraMargin times further out.
  let fit = min(fbW / rw, fbH / rh)
  result.s = if exact: fit else: fit / CameraMargin
  let rwF = float(rw) * result.s
  let rhF = float(rh) * result.s
  result.left = int(floor((float(fbW) - rwF) / 2 + 0.5))
  result.top = int(floor((float(fbH) - rhF) / 2 + 0.5))
  result.rwS = max(1, int(floor(rwF + 0.5)))
  result.rhS = max(1, int(floor(rhF + 0.5)))

const WhiteBelowLuminance = 0.17913   ## sqrt(1.05 * 0.05) - 0.05: below it white has more contrast than black

var srgbLinear: array[256, float32]

proc initLuts() =
  for i in 0 .. 255:
    let c = float(i) / 255.0
    srgbLinear[i] = float32(if c <= 0.04045: c / 12.92 else: pow((c + 0.055) / 1.055, 2.4))

proc outlineIsWhite(r, g, b: int): bool =
  ## Does white give more contrast (WCAG contrast ratio) than black against this colour?
  let lum = 0.2126 * srgbLinear[r] + 0.7152 * srgbLinear[g] + 0.0722 * srgbLinear[b]
  lum < WhiteBelowLuminance

# ---------------------------------------------------------------------------------------------
# Images on disk: probing, converting with vips, the cache and its pyramid
# ---------------------------------------------------------------------------------------------
type
  ImageInfo = object
    w, h, bands: int
    format, interp: string

  LevelMap = object
    w, h: int
    mf: MemFile
    mapped: bool

  Image = ref object
    srcPath: string
    dir: string                 ## cache folder
    w, h, bands: int
    levels: seq[LevelMap]       ## [0] full size; the others appear as the pyramid is built
    levelCount: int             ## how many levels the pyramid has when finished

proc vipsTool(name: string): string =
  let exeName = addFileExt(name, ExeExt)
  if VipsDir.len > 0 and fileExists(VipsDir / exeName): return VipsDir / exeName
  findExe(name)

proc runBlocking(exe: string, args: seq[string]): tuple[code: int, output: string] =
  var p = startProcess(exe, args = args, options = {poStdErrToStdOut, poDaemon})
  result.output = p.outputStream.readAll()
  result.code = p.waitForExit()
  p.close()

proc parseHeaderOutput(text: string): tuple[ok: bool, info: ImageInfo] =
  ## The text `vipsheader -f width -f height -f bands -f format -f interpretation file` prints.
  var lines: seq[string]
  for l in text.splitLines:
    if l.strip.len > 0: lines.add l.strip
  if lines.len < 5: return
  try:
    result.info.w = parseInt(lines[^5])
    result.info.h = parseInt(lines[^4])
    result.info.bands = parseInt(lines[^3])
    result.info.format = lines[^2]
    result.info.interp = lines[^1]
    result.ok = result.info.w > 0 and result.info.h > 0 and result.info.bands > 0
  except ValueError:
    result.ok = false

proc needsConversion(info: ImageInfo): bool =
  ## False when vips can write the pixels as they are: 8-bit, 3 or 4 bands, sRGB-like.
  not (info.format.contains("UCHAR") and info.bands in 3 .. 4 and
       (info.interp.contains("_sRGB") or info.interp.contains("_MULTIBAND")))

proc probeImage(path: string): tuple[info: ImageInfo, error: string] =
  let tool = vipsTool("vipsheader")
  if tool.len == 0:
    result.error = "vipsheader was not found: set VipsDir at the top of wallcrop.nim or put vips on PATH"
    return
  let (code, text) = runBlocking(tool, @["-f", "width", "-f", "height", "-f", "bands", "-f", "format",
                                        "-f", "interpretation", path])
  let parsed = parseHeaderOutput(text)
  if code != 0 or not parsed.ok:
    let first = text.strip.splitLines
    result.error = "vips cannot read this file" & (if first.len > 0 and first[0].len > 0: ": " & first[0] else: "")
    return
  result.info = parsed.info

proc levelFile(dir: string, level: int): string = dir / ("l" & $level & ".raw")

proc cacheDirFor(srcPath: string): string =
  let full = absolutePath(srcPath)
  var sig = full
  try: sig &= "|" & $getFileSize(full) & "|" & $getLastModificationTime(full).toUnix
  except OSError: discard
  cacheRoot() / (safeStem(splitFile(full).name) & "-" & toHex(fnv64(sig), 16).toLowerAscii)

proc metaText(srcPath: string, w, h, bands: int): string =
  let full = absolutePath(srcPath)
  "wallcrop-cache " & $CacheVersion & "\nsource " & full & "\nsize " & $getFileSize(full) & "\nmtime " &
    $getLastModificationTime(full).toUnix & "\nwidth " & $w & "\nheight " & $h & "\nbands " & $bands & "\n"

proc mapLevel(img: Image, level: int): bool =
  ## Memory-map level `level` if its file is complete.
  if level >= img.levelCount: return false
  while img.levels.len <= level: img.levels.add LevelMap()
  if img.levels[level].mapped: return true
  let (lw, lh) = levelSize(img.w, img.h, level)
  let path = levelFile(img.dir, level)
  if not fileExists(path) or getFileSize(path) != int64(lw) * lh * img.bands: return false
  try:
    img.levels[level].mf = memfiles.open(path, fmRead)
    img.levels[level].w = lw
    img.levels[level].h = lh
    img.levels[level].mapped = true
    result = true
  except OSError, IOError:
    result = false

proc pixelsOf(img: Image, level: int): Bytes {.inline.} = cast[Bytes](img.levels[level].mf.mem)

proc openCached(srcPath, dir: string): Image =
  ## The image from a finished cache folder, or nil.
  let metaPath = dir / "meta.txt"
  if not fileExists(metaPath) or not fileExists(levelFile(dir, 0)): return nil
  var w, h, bands = 0
  var ok = false
  try:
    let full = absolutePath(srcPath)
    var sizeOk, mtimeOk, versionOk = false
    for line in readFile(metaPath).splitLines:
      let parts = line.split(' ', 1)
      if parts.len < 2: continue
      case parts[0]
      of "wallcrop-cache": versionOk = parts[1] == $CacheVersion
      of "size": sizeOk = parts[1] == $getFileSize(full)
      of "mtime": mtimeOk = parts[1] == $getLastModificationTime(full).toUnix
      of "width": w = parseInt(parts[1])
      of "height": h = parseInt(parts[1])
      of "bands": bands = parseInt(parts[1])
      else: discard
    ok = versionOk and sizeOk and mtimeOk and w > 0 and h > 0 and bands in 3 .. 4
  except CatchableError:
    ok = false
  if not ok: return nil
  result = Image(srcPath: absolutePath(srcPath), dir: dir, w: w, h: h, bands: bands,
                 levelCount: levelCountFor(w, h))
  if not mapLevel(result, 0): return nil
  var l = 1
  while mapLevel(result, l): inc l

proc closeImage(img: Image) =
  if img == nil: return
  for lv in img.levels.mitems:
    if lv.mapped:
      lv.mf.close()
      lv.mapped = false

type
  PrepState = enum psRunning, psDone, psFailed
  Prep = ref object
    src, dir, part: string
    info: ImageInfo
    process: Process
    started: MonoTime
    state: PrepState
    error: string
    image: Image

proc startPrep(src: string, info: ImageInfo): Prep =
  ## Start vips converting `src` into raw pixels in its cache folder (returns at once).
  let dir = cacheDirFor(src)
  createDir(dir)
  let part = dir / "l0.part.raw"
  if fileExists(part): removeFile(part)
  let exe = vipsTool("vips")
  result = Prep(src: absolutePath(src), dir: dir, part: part, info: info, started: getMonoTime())
  if exe.len == 0:
    result.state = psFailed
    result.error = "vips was not found: set VipsDir at the top of wallcrop.nim or put vips on PATH"
    return
  let input = absolutePath(src) & "[access=sequential]"
  let args = if needsConversion(info): @["colourspace", input, part, "srgb"] else: @["copy", input, part]
  result.process = startProcess(exe, args = args, options = {poStdErrToStdOut, poDaemon})
  result.state = psRunning

proc pollPrep(p: Prep) =
  ## If the conversion finished, check it and open the image (sets state, image or error).
  if p.state != psRunning or p.process.running: return
  let text = p.process.outputStream.readAll()
  let code = p.process.waitForExit()
  p.process.close()
  if code != 0:
    p.state = psFailed
    let l = text.strip.splitLines
    p.error = "vips failed" & (if l.len > 0 and l[0].len > 0: ": " & l[0] else: "")
    return
  try:
    let size = getFileSize(p.part)
    let pixelCount = int64(p.info.w) * p.info.h
    let bands = int(size div pixelCount)
    if size != pixelCount * bands or bands notin 3 .. 4:
      p.state = psFailed
      p.error = "unexpected converted size (" & $size & " bytes for " & $p.info.w & " x " & $p.info.h & ")"
      return
    moveFile(p.part, levelFile(p.dir, 0))
    writeFile(p.dir / "meta.txt", metaText(p.src, p.info.w, p.info.h, bands))
    p.image = openCached(p.src, p.dir)
    if p.image == nil:
      p.state = psFailed
      p.error = "the converted image could not be opened"
    else:
      p.state = psDone
  except CatchableError as e:
    p.state = psFailed
    p.error = e.msg

proc cancelPrep(p: Prep) =
  if p == nil or p.state != psRunning: return
  try:
    p.process.terminate()
    discard p.process.waitForExit()
    p.process.close()
  except CatchableError: discard
  p.state = psFailed
  p.error = "cancelled"
  try:
    if fileExists(p.part): removeFile(p.part)
  except CatchableError: discard

proc openBlocking(src: string): tuple[image: Image, error: string, cached: bool] =
  ## Open an image, converting it first if it has no cache yet (waits for vips).
  let dir = cacheDirFor(src)
  result.image = openCached(src, dir)
  if result.image != nil:
    result.cached = true
    return
  let (info, err) = probeImage(src)
  if err.len > 0:
    result.error = err
    return
  let prep = startPrep(src, info)
  while prep.state == psRunning:
    pollPrep(prep)
    sleep(40)
  if prep.state == psFailed: result.error = prep.error else: result.image = prep.image

# --- the pyramid, built on a thread -----------------------------------------------------------
type
  PyrJob = object
    dir: cstring
    w, h, bands, firstLevel, levelCount: int
    cancel: Atomic[bool]
    ready: Atomic[int]        ## levels complete, level 0 included
    finished: Atomic[bool]

  Pyramid = ref object
    job: ptr PyrJob
    thread: Thread[ptr PyrJob]

proc buildLevel(dir: string, level, bands, sw, sh: int, cancel: ptr Atomic[bool]): bool =
  let dw = halfCeil(sw)
  let dh = halfCeil(sh)
  let finalPath = levelFile(dir, level)
  let partPath = finalPath & ".part"
  if fileExists(partPath): removeFile(partPath)
  var src = memfiles.open(levelFile(dir, level - 1), fmRead)
  var dst = memfiles.open(partPath, fmReadWrite, newFileSize = dw * dh * bands)
  var ctx = LevelCtx(src: cast[Bytes](src.mem), dst: cast[Bytes](dst.mem), sw: sw, sh: sh, dw: dw,
                     dh: dh, bands: bands)
  parallelFor(dh, 16, addr ctx, levelRows, cancel)
  src.close()
  dst.close()
  if cancel[].load(moRelaxed):
    removeFile(partPath)
    return false
  moveFile(partPath, finalPath)
  true

proc pyramidMain(job: ptr PyrJob) {.thread.} =
  let dir = $job.dir
  var (sw, sh) = levelSize(job.w, job.h, job.firstLevel - 1)
  try:
    for level in job.firstLevel ..< job.levelCount:
      if job.cancel.load(moRelaxed): break
      if not buildLevel(dir, level, job.bands, sw, sh, addr job.cancel): break
      sw = halfCeil(sw)
      sh = halfCeil(sh)
      job.ready.store(level + 1)
  except CatchableError:
    discard
  job.finished.store(true)

proc startPyramid(img: Image): Pyramid =
  ## Build the missing levels in the background.  nil when the pyramid is already complete.
  var first = 1
  while first < img.levelCount and fileExists(levelFile(img.dir, first)) and
        getFileSize(levelFile(img.dir, first)) == int64(levelSize(img.w, img.h, first)[0]) *
        levelSize(img.w, img.h, first)[1] * img.bands:
    inc first
  if first >= img.levelCount: return nil
  result = Pyramid()
  result.job = cast[ptr PyrJob](allocShared0(sizeof(PyrJob)))
  result.job.dir = cast[cstring](allocShared0(img.dir.len + 1))
  copyMem(result.job.dir, unsafeAddr img.dir[0], img.dir.len)
  result.job.w = img.w
  result.job.h = img.h
  result.job.bands = img.bands
  result.job.firstLevel = first
  result.job.levelCount = img.levelCount
  result.job.ready.store(first)
  createThread(result.thread, pyramidMain, result.job)

proc stopPyramid(p: Pyramid) =
  if p == nil: return
  p.job.cancel.store(true)
  joinThread(p.thread)
  deallocShared(p.job.dir)
  deallocShared(p.job)

proc pyramidDone(p: Pyramid): bool = p == nil or p.job.finished.load()

# ---------------------------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------------------------
proc exportName(stem: string, x0, y0, outW, outH, n: int, mode: SampleMode): string =
  safeStem(stem) & "_" & $outW & "x" & $outH & "_at_" & $x0 & "_" & $y0 &
    (if n == 1: "_1x1" else: "_" & $n & "x" & $n & "_" & ModeFileNames[mode]) & ".png"

proc uniquePath(dir, fileName: string): string =
  ## dir/fileName, or with _2, _3 ... before the extension when it exists (never overwrites).
  result = dir / fileName
  var i = 2
  let (_, name, ext) = splitFile(fileName)
  while fileExists(result):
    result = dir / (name & "_" & $i & ext)
    inc i

proc writePngWithVips(rawFile: string, w, h, bands: int, outPath: string): string =
  ## Turn a file of w x h x bands raw bytes into a PNG with vips.  Returns "" or an error.
  let exe = vipsTool("vips")
  if exe.len == 0: return "vips was not found"
  let (code, text) = runBlocking(exe, @["rawload", rawFile, outPath, $w, $h, $bands])
  if code != 0 or not fileExists(outPath):
    let l = text.strip.splitLines
    return "vips could not write the PNG" & (if l.len > 0 and l[0].len > 0: ": " & l[0] else: "")

proc exportPixelsToPng*(pixels: Bytes, srcW, srcH, bands, x0, y0, outW, outH, n: int, mode: SampleMode,
                        outPath: string, rowsDone: ptr Atomic[int] = nil,
                        cancel: ptr Atomic[bool] = nil): string =
  ## Export the n*outW x n*outH rectangle at (x0, y0) of the srcW x srcH x bands pixels as a PNG.
  ## Returns "" or an error message.  (Safe to run on a thread: it touches no shared Nim data.)
  if x0 < 0 or y0 < 0 or x0 + n * outW > srcW or y0 + n * outH > srcH:
    return "the rectangle is not inside the image"
  let total = int64(outW) * outH * bands
  let buf = cast[Bytes](allocShared(int(total)))
  try:
    exportBlocks(pixels, srcW, bands, x0, y0, outW, outH, n, mode, buf, rowsDone, cancel)
    if cancel != nil and cancel[].load(moRelaxed): return "cancelled"
    createDir(appDir())
    let tmp = appDir() / ("export-" & $getCurrentProcessId() & ".raw")
    var f: File
    if not f.open(tmp, fmWrite): return "cannot write " & tmp
    let written = writeBuffer(f, buf, int(total))
    f.close()
    if written != int(total): return "cannot write " & tmp & " (disk full?)"
    result = writePngWithVips(tmp, outW, outH, bands, outPath)
    try: removeFile(tmp)
    except OSError: discard
  except CatchableError as e:
    result = "export failed: " & e.msg
  finally:
    deallocShared(buf)

proc exportToPng*(img: Image, x0, y0, outW, outH, n: int, mode: SampleMode, outPath: string,
                  rowsDone: ptr Atomic[int] = nil, cancel: ptr Atomic[bool] = nil): string =
  exportPixelsToPng(pixelsOf(img, 0), img.w, img.h, img.bands, x0, y0, outW, outH, n, mode, outPath,
                    rowsDone, cancel)

# ---------------------------------------------------------------------------------------------
# The pixel font and the software framebuffer (everything on screen is drawn here, on the CPU)
# ---------------------------------------------------------------------------------------------
const GlyphSource = [
  ('A', "01110/10001/10001/11111/10001/10001/00000"), ('B', "11110/10001/11110/10001/10001/11110/00000"),
  ('C', "01110/10001/10000/10000/10001/01110/00000"), ('D', "11110/10001/10001/10001/10001/11110/00000"),
  ('E', "11111/10000/11110/10000/10000/11111/00000"), ('F', "11111/10000/11110/10000/10000/10000/00000"),
  ('G', "01110/10000/10000/10111/10001/01110/00000"), ('H', "10001/10001/11111/10001/10001/10001/00000"),
  ('I', "11111/00100/00100/00100/00100/11111/00000"), ('J', "00111/00010/00010/00010/10010/01100/00000"),
  ('K', "10001/10010/11100/10010/10001/10001/00000"), ('L', "10000/10000/10000/10000/10000/11111/00000"),
  ('M', "10001/11011/10101/10001/10001/10001/00000"), ('N', "10001/11001/10101/10011/10001/10001/00000"),
  ('O', "01110/10001/10001/10001/10001/01110/00000"), ('P', "11110/10001/10001/11110/10000/10000/00000"),
  ('Q', "01110/10001/10001/10101/10010/01101/00000"), ('R', "11110/10001/10001/11110/10100/10010/00000"),
  ('S', "01111/10000/01110/00001/00001/11110/00000"), ('T', "11111/00100/00100/00100/00100/00100/00000"),
  ('U', "10001/10001/10001/10001/10001/01110/00000"), ('V', "10001/10001/10001/10001/01010/00100/00000"),
  ('W', "10001/10001/10001/10101/10101/01010/00000"), ('X', "10001/01010/00100/00100/01010/10001/00000"),
  ('Y', "10001/10001/01010/00100/00100/00100/00000"), ('Z', "11111/00010/00100/01000/10000/11111/00000"),
  ('a', "00000/01110/10001/01111/10001/01111/00000"), ('b', "10000/10000/11110/10001/10001/11110/00000"),
  ('c', "00000/00000/01110/10000/10000/01110/00000"), ('d', "00001/00001/01111/10001/10001/01111/00000"),
  ('e', "00000/01111/10000/11110/10000/01111/00000"), ('f', "00110/01000/11110/01000/01000/01000/00000"),
  ('g', "00000/01111/10001/10001/01111/00001/01110"), ('h', "10000/10000/11110/10001/10001/10001/00000"),
  ('i', "00000/00100/00000/00100/00100/00100/00000"), ('j', "00000/00010/00000/00010/00010/10010/01100"),
  ('k', "10000/10000/10010/10100/11100/10010/00000"), ('l', "10000/10000/10000/10000/10000/01111/00000"),
  ('m', "00000/11011/10101/10101/10101/10101/00000"), ('n', "00000/11010/10110/10010/10010/10010/00000"),
  ('o', "00000/01110/10001/10001/10001/01110/00000"), ('p', "00000/11110/10001/11110/10000/10000/00000"),
  ('q', "00000/01111/10001/01111/00001/00001/00000"), ('r', "00000/11110/10001/11110/10100/10010/00000"),
  ('s', "00000/01111/10000/01110/00001/11110/00000"), ('t', "01000/01000/11110/01000/01000/00110/00000"),
  ('u', "00000/10001/10001/10001/10001/01111/00000"), ('v', "00000/10001/10001/10001/01010/00100/00000"),
  ('w', "00000/10001/10001/10101/10101/01010/00000"), ('x', "00000/10001/01010/00100/01010/10001/00000"),
  ('y', "00000/10001/10001/01111/00001/00001/01110"), ('z', "00000/11111/00010/00100/01000/11111/00000"),
  ('0', "00000/01110/10011/10101/11001/01110/00000"), ('1', "00000/00100/01100/00100/00100/01110/00000"),
  ('2', "00000/01110/10001/00110/01000/11111/00000"), ('3', "00000/11110/00001/00110/00001/11110/00000"),
  ('4', "00000/00110/01010/10010/11111/00010/00000"), ('5', "00000/11111/10000/11110/00001/01110/00000"),
  ('6', "00000/01110/10000/11110/10001/01110/00000"), ('7', "00000/11111/00001/00010/00100/00100/00000"),
  ('8', "00000/01110/10001/01110/10001/01110/00000"), ('9', "00000/01110/10001/01111/00001/01110/00000"),
  (' ', "00000/00000/00000/00000/00000/00000/00000"), ('[', "01110/01000/01000/01000/01000/01000/01110"),
  (']', "01110/00010/00010/00010/00010/00010/01110"), ('(', "00100/01000/10000/10000/10000/01000/00100"),
  (')', "00100/00010/00001/00001/00001/00010/00100"), ('+', "00000/00100/00100/11111/00100/00100/00000"),
  ('-', "00000/00000/00000/11111/00000/00000/00000"), ('/', "00001/00010/00100/00100/01000/10000/00000"),
  (':', "00000/00100/00000/00000/00100/00000/00000"), ('.', "00000/00000/00000/00000/00000/00100/00000"),
  (',', "00000/00000/00000/00000/00100/00100/01000"), (';', "00000/00100/00000/00000/00100/00100/01000"),
  ('>', "00000/10000/01000/00100/01000/10000/00000"), ('<', "00000/00001/00010/00100/00010/00001/00000"),
  ('|', "00100/00100/00100/00100/00100/00100/00100"), ('_', "00000/00000/00000/00000/00000/00000/11111"),
  ('=', "00000/00000/11111/00000/11111/00000/00000"), ('*', "00000/10101/01110/11111/01110/10101/00000"),
  ('%', "11001/11010/00010/00100/01000/01011/10011"), ('!', "00100/00100/00100/00100/00100/00000/00100"),
  ('?', "01110/10001/00001/00110/00100/00000/00100"), ('\\', "10000/01000/00100/00100/00010/00001/00000"),
  ('"', "01010/01010/00000/00000/00000/00000/00000"), ('\'', "01000/01000/00000/00000/00000/00000/00000"),
  ('~', "00000/00000/01000/10101/00010/00000/00000"), ('@', "01110/10001/10111/10101/10110/10000/01110"),
  ('#', "01010/01010/11111/01010/11111/01010/01010"), ('^', "00100/01010/10001/00000/00000/00000/00000"),
  ('$', "00100/01111/10100/01110/00101/11110/00100"), ('&', "01100/10010/10100/01000/10101/10010/01101"),
  ('`', "01000/00100/00000/00000/00000/00000/00000"), ('{', "00110/00100/00100/01000/00100/00100/00110"),
  ('}', "01100/00100/00100/00010/00100/00100/01100")]

var glyphBits: array[256, array[7, uint8]]
var glyphKnown: array[256, bool]

proc initFont() =
  for (c, rows) in GlyphSource:
    let parts = rows.split('/')
    doAssert parts.len == 7, "bad glyph " & $c
    glyphKnown[ord(c)] = true
    for r in 0 ..< 7:
      doAssert parts[r].len == 5, "bad glyph row " & $c
      var bits = 0'u8
      for col in 0 ..< 5:
        if parts[r][col] == '1': bits = bits or (1'u8 shl (4 - col))
      glyphBits[ord(c)][r] = bits

iterator glyphsOf(s: string): char =
  ## The characters to draw: a UTF-8 sequence that the font has no letter for is one '?'.
  var i = 0
  while i < s.len:
    if ord(s[i]) < 0x80:
      yield s[i]
      inc i
    else:
      yield '?'
      inc i
      while i < s.len and (ord(s[i]) and 0xC0) == 0x80: inc i

proc textWidth(s: string, sc: int): int =
  var count = 0
  for _ in glyphsOf(s): inc count
  if count == 0: 0 else: count * 6 * sc - sc

type Fb = object
  w, h: int
  px: seq[uint32]

proc resizeFb(fb: var Fb, w, h: int) =
  if fb.w != w or fb.h != h:
    fb.w = w
    fb.h = h
    fb.px = newSeq[uint32](w * h)

proc fillRect(fb: var Fb, x0, y0, x1, y1: int, color: uint32) =
  ## [x0, x1) x [y0, y1), clipped
  let a = max(0, x0)
  let b = min(fb.w, x1)
  for y in max(0, y0) ..< min(fb.h, y1):
    for x in a ..< b: fb.px[y * fb.w + x] = color

proc dimRect(fb: var Fb, x0, y0, x1, y1: int) =
  ## Half-transparent black: every colour channel halved.
  let a = max(0, x0)
  let b = min(fb.w, x1)
  for y in max(0, y0) ..< min(fb.h, y1):
    for x in a ..< b:
      let i = y * fb.w + x
      fb.px[i] = 0xFF000000'u32 or ((fb.px[i] shr 1) and 0x007F7F7F'u32)

const White = 0xFFFFFFFF'u32
const Black = 0xFF000000'u32

proc drawText(fb: var Fb, x, y: int, s: string, sc: int) =
  var cx = x
  for c in glyphsOf(s):
    let bits = glyphBits[if glyphKnown[ord(c)]: ord(c) else: ord('?')]
    for r in 0 ..< 7:
      for col in 0 ..< 5:
        if (bits[r] and (1'u8 shl (4 - col))) != 0:
          fb.fillRect(cx + col * sc, y + r * sc, cx + (col + 1) * sc, y + (r + 1) * sc, White)
    cx += 6 * sc

proc tailGap(sc: int): int = 8 * sc

proc drawPanel(fb: var Fb, x, y: int, lines: seq[string], sc: int, selected = -1,
               tails: seq[string] = @[], tailSc = 0): tuple[w, h: int] =
  ## White text on a half-transparent black backdrop; the selected line gets a darker bar.  A line may have a
  ## "tail" (tails[i], drawn after it in the smaller size tailSc, its bottom lined up with the line's).
  let pad = 2 * sc
  let lineH = 9 * sc
  proc tail(i: int): string = (if i < tails.len: tails[i] else: "")
  var w = 0
  for i, l in lines:
    let t = tail(i)
    w = max(w, textWidth(l, sc) + (if t.len > 0: tailGap(sc) + textWidth(t, tailSc) else: 0))
  result.w = w + 2 * pad
  result.h = lines.len * lineH + 2 * pad - 2 * sc
  fb.dimRect(x, y, x + result.w, y + result.h)
  if selected >= 0 and selected < lines.len:
    let top = y + pad + selected * lineH - sc
    fb.dimRect(x, top, x + result.w, top + lineH)
  for i, l in lines:
    let top = y + pad + i * lineH
    fb.drawText(x + pad, top, l, sc)
    let t = tail(i)
    if t.len > 0:
      fb.drawText(x + pad + textWidth(l, sc) + tailGap(sc), top + 7 * sc - 7 * tailSc, t, tailSc)

# ---------------------------------------------------------------------------------------------
# SDL2, loaded at run time (so the program starts - and the self-test runs - without it)
# ---------------------------------------------------------------------------------------------
type
  SdlRect = object
    x, y, w, h: cint
  SdlEvent = object
    data: array[7, uint64]            ## SDL_Event is a 56-byte union; fields are read at their offsets

var
  sdlInit: proc (flags: uint32): cint {.cdecl.}
  sdlQuit: proc () {.cdecl.}
  sdlGetError: proc (): cstring {.cdecl.}
  sdlSetHint: proc (name, value: cstring): cint {.cdecl.}
  sdlCreateWindow: proc (title: cstring, x, y, w, h: cint, flags: uint32): pointer {.cdecl.}
  sdlDestroyWindow: proc (win: pointer) {.cdecl.}
  sdlSetWindowTitle: proc (win: pointer, title: cstring) {.cdecl.}
  sdlSetWindowFullscreen: proc (win: pointer, flags: uint32): cint {.cdecl.}
  sdlSetWindowMinimumSize: proc (win: pointer, w, h: cint) {.cdecl.}
  sdlGetDisplayBounds: proc (index: cint, r: ptr SdlRect): cint {.cdecl.}
  sdlCreateRenderer: proc (win: pointer, index: cint, flags: uint32): pointer {.cdecl.}
  sdlDestroyRenderer: proc (r: pointer) {.cdecl.}
  sdlGetRendererOutputSize: proc (r: pointer, w, h: ptr cint): cint {.cdecl.}
  sdlCreateTexture: proc (r: pointer, format: uint32, access, w, h: cint): pointer {.cdecl.}
  sdlDestroyTexture: proc (t: pointer) {.cdecl.}
  sdlUpdateTexture: proc (t: pointer, rect: ptr SdlRect, pixels: pointer, pitch: cint): cint {.cdecl.}
  sdlRenderCopy: proc (r, t: pointer, src, dst: ptr SdlRect): cint {.cdecl.}
  sdlRenderPresent: proc (r: pointer) {.cdecl.}
  sdlWaitEventTimeout: proc (e: ptr SdlEvent, ms: cint): cint {.cdecl.}
  sdlPollEvent: proc (e: ptr SdlEvent): cint {.cdecl.}
  sdlStartTextInput: proc () {.cdecl.}
  sdlStopTextInput: proc () {.cdecl.}
  sdlFree: proc (p: pointer) {.cdecl.}

proc findSdl(): tuple[lib: LibHandle, tried: seq[string]] =
  var candidates: seq[string]
  if SdlLibrary.len > 0: candidates.add SdlLibrary
  when defined(windows):
    let name = "SDL2.dll"
    candidates.add getAppDir() / name
    candidates.add parentDir(SourceFile) / name
    candidates.add appDir() / name
  else:
    let name = "libSDL2-2.0.so.0"
    candidates.add getAppDir() / name
    candidates.add "libSDL2-2.0.so"
    candidates.add "libSDL2.so"
  candidates.add name
  for c in candidates:
    result.tried.add c
    result.lib = loadLib(c)
    if result.lib != nil: return

proc loadSdl(): string =
  ## Load SDL2 and every function the program uses.  Returns "" or what went wrong.
  let (lib, tried) = findSdl()
  if lib == nil:
    return "SDL2 could not be loaded (a 64-bit SDL2 is needed). Looked for:\n  " & tried.join("\n  ") &
      "\nPut SDL2.dll in one of those folders, or set SdlLibrary at the top of wallcrop.nim."
  template bindFn(v: untyped, name: string) =
    v = cast[typeof(v)](symAddr(lib, name))
    if v == nil: return "this SDL2 lacks " & name & " (it is too old?)"
  bindFn(sdlInit, "SDL_Init")
  bindFn(sdlQuit, "SDL_Quit")
  bindFn(sdlGetError, "SDL_GetError")
  bindFn(sdlSetHint, "SDL_SetHint")
  bindFn(sdlCreateWindow, "SDL_CreateWindow")
  bindFn(sdlDestroyWindow, "SDL_DestroyWindow")
  bindFn(sdlSetWindowTitle, "SDL_SetWindowTitle")
  bindFn(sdlSetWindowFullscreen, "SDL_SetWindowFullscreen")
  bindFn(sdlSetWindowMinimumSize, "SDL_SetWindowMinimumSize")
  bindFn(sdlGetDisplayBounds, "SDL_GetDisplayBounds")
  bindFn(sdlCreateRenderer, "SDL_CreateRenderer")
  bindFn(sdlDestroyRenderer, "SDL_DestroyRenderer")
  bindFn(sdlGetRendererOutputSize, "SDL_GetRendererOutputSize")
  bindFn(sdlCreateTexture, "SDL_CreateTexture")
  bindFn(sdlDestroyTexture, "SDL_DestroyTexture")
  bindFn(sdlUpdateTexture, "SDL_UpdateTexture")
  bindFn(sdlRenderCopy, "SDL_RenderCopy")
  bindFn(sdlRenderPresent, "SDL_RenderPresent")
  bindFn(sdlWaitEventTimeout, "SDL_WaitEventTimeout")
  bindFn(sdlPollEvent, "SDL_PollEvent")
  bindFn(sdlStartTextInput, "SDL_StartTextInput")
  bindFn(sdlStopTextInput, "SDL_StopTextInput")
  bindFn(sdlFree, "SDL_free")

const
  SdlInitVideo = 0x20'u32
  SdlWindowShown = 0x4'u32
  SdlWindowResizable = 0x20'u32
  SdlWindowHighDpi = 0x2000'u32
  SdlWindowFullscreenDesktop = 0x1001'u32
  SdlRendererAccelerated = 0x2'u32
  SdlRendererVsync = 0x4'u32
  SdlPixelFormatArgb8888 = 0x16362004'u32
  SdlTextureStreaming = 1
  SdlWindowPosCentered = 0x2FFF0000
  EvQuit = 0x100'u32
  EvWindow = 0x200'u32
  EvKeyDown = 0x300'u32
  EvKeyUp = 0x301'u32
  EvTextInput = 0x303'u32
  EvDropFile = 0x1000'u32
  WinFocusLost = 13'u8

proc u32At(e: SdlEvent, off: int): uint32 =
  copyMem(addr result, cast[pointer](cast[uint](unsafeAddr e) + uint(off)), 4)

proc i32At(e: SdlEvent, off: int): int32 =
  copyMem(addr result, cast[pointer](cast[uint](unsafeAddr e) + uint(off)), 4)

proc u16At(e: SdlEvent, off: int): uint16 =
  copyMem(addr result, cast[pointer](cast[uint](unsafeAddr e) + uint(off)), 2)

proc u8At(e: SdlEvent, off: int): uint8 = cast[ptr uint8](cast[uint](unsafeAddr e) + uint(off))[]

proc ptrAt(e: SdlEvent, off: int): pointer =
  copyMem(addr result, cast[pointer](cast[uint](unsafeAddr e) + uint(off)), sizeof(pointer))

proc textAt(e: SdlEvent, off: int): string =
  ## the NUL-terminated text starting at byte `off` of the event
  var i = off
  while i < 56 and u8At(e, i) != 0:
    result.add char(u8At(e, i))
    inc i

proc makeDpiAware() =
  ## Windows: draw at the monitor's real pixels (otherwise Windows stretches the window and every
  ## pixel the crop shows would be blurred).
  when defined(windows):
    let user32 = loadLib("user32.dll")
    if user32 != nil:
      let perMonitor = cast[proc (ctx: int): int32 {.stdcall.}](symAddr(user32, "SetProcessDpiAwarenessContext"))
      if perMonitor != nil:
        discard perMonitor(-4)                     # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
      else:
        let system = cast[proc (): int32 {.stdcall.}](symAddr(user32, "SetProcessDPIAware"))
        if system != nil: discard system()

# ---------------------------------------------------------------------------------------------
# File and folder dialogs
# ---------------------------------------------------------------------------------------------
when defined(windows):
  type
    OpenFileName = object
      lStructSize: int32
      hwndOwner: pointer
      hInstance: pointer
      lpstrFilter: ptr uint16
      lpstrCustomFilter: ptr uint16
      nMaxCustFilter: int32
      nFilterIndex: int32
      lpstrFile: ptr uint16
      nMaxFile: int32
      lpstrFileTitle: ptr uint16
      nMaxFileTitle: int32
      lpstrInitialDir: ptr uint16
      lpstrTitle: ptr uint16
      flags: int32
      nFileOffset: uint16
      nFileExtension: uint16
      lpstrDefExt: ptr uint16
      lCustData: int
      lpfnHook: pointer
      lpTemplateName: ptr uint16
      pvReserved: pointer
      dwReserved: int32
      flagsEx: int32
    BrowseInfo = object
      hwndOwner: pointer
      pidlRoot: pointer
      pszDisplayName: ptr uint16
      lpszTitle: ptr uint16
      ulFlags: uint32
      lpfn: pointer
      lParam: int
      iImage: int32

  proc getOpenFileNameW(ofn: ptr OpenFileName): int32 {.stdcall, dynlib: "comdlg32.dll", importc: "GetOpenFileNameW".}
  proc shBrowseForFolderW(bi: ptr BrowseInfo): pointer {.stdcall, dynlib: "shell32.dll", importc: "SHBrowseForFolderW".}
  proc shGetPathFromIDListW(pidl: pointer, path: ptr uint16): int32 {.stdcall, dynlib: "shell32.dll", importc: "SHGetPathFromIDListW".}
  proc coTaskMemFree(p: pointer) {.stdcall, dynlib: "ole32.dll", importc: "CoTaskMemFree".}
  proc coInitializeEx(reserved: pointer, flags: uint32): int32 {.stdcall, dynlib: "ole32.dll", importc: "CoInitializeEx".}
  proc sendMessageW(h: pointer, msg: uint32, wp: int, lp: int): int {.stdcall, dynlib: "user32.dll", importc: "SendMessageW".}

  proc toWide(s: string): seq[uint16] =
    for r in s.runes:
      let c = int(r)
      if c < 0x10000: result.add uint16(c)
      else:
        let v = c - 0x10000
        result.add uint16(0xD800 + (v shr 10))
        result.add uint16(0xDC00 + (v and 0x3FF))
    result.add 0'u16

  proc fromWide(p: ptr uint16): string =
    var units: seq[uint16]
    var i = 0
    while cast[ptr UncheckedArray[uint16]](p)[i] != 0:
      units.add cast[ptr UncheckedArray[uint16]](p)[i]
      inc i
    var k = 0
    while k < units.len:
      var c = int(units[k])
      if c in 0xD800 .. 0xDBFF and k + 1 < units.len:
        c = 0x10000 + ((c - 0xD800) shl 10) + (int(units[k + 1]) - 0xDC00)
        inc k
      result.add toUTF8(Rune(c))
      inc k

  proc browseCallback(hwnd: pointer, msg: uint32, lp: int, data: int): int32 {.stdcall.} =
    if msg == 1:                                    # BFFM_INITIALIZED: select the starting folder
      discard sendMessageW(hwnd, 0x467, 1, data)    # BFFM_SETSELECTIONW
    0

  proc pickFile(initialDir: string): string =
    var buffer = newSeq[uint16](32768)
    let filter = toWide("Images\0*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.webp;*.jxl;*.avif;*.heic;*.bmp;*.gif;*.exr;*.hdr;*.ppm;*.pgm;*.jp2\0All files\0*.*\0")
    var title = toWide("Open an image")
    var dir = toWide(initialDir)
    var ofn: OpenFileName
    ofn.lStructSize = int32(sizeof(OpenFileName))
    ofn.lpstrFilter = unsafeAddr filter[0]
    ofn.nFilterIndex = 1
    ofn.lpstrFile = addr buffer[0]
    ofn.nMaxFile = int32(buffer.len)
    if initialDir.len > 0: ofn.lpstrInitialDir = addr dir[0]
    ofn.lpstrTitle = addr title[0]
    ofn.flags = 0x1000 or 0x800 or 0x80000 or 0x8 or 0x4   # file+path must exist, explorer style, keep cwd, no read-only box
    if getOpenFileNameW(addr ofn) != 0: result = fromWide(addr buffer[0])

  proc pickFolder(initialDir: string): string =
    discard coInitializeEx(nil, 2'u32)
    var display = newSeq[uint16](1024)
    var title = toWide("Choose the folder to save the exported images in")
    var start = toWide(initialDir)
    var bi: BrowseInfo
    bi.pszDisplayName = addr display[0]
    bi.lpszTitle = addr title[0]
    bi.ulFlags = 0x1 or 0x40 or 0x10                         # folders only, new style, with an edit box
    if initialDir.len > 0:
      bi.lpfn = cast[pointer](browseCallback)
      bi.lParam = cast[int](addr start[0])
    let pidl = shBrowseForFolderW(addr bi)
    if pidl != nil:
      var path = newSeq[uint16](32768)
      if shGetPathFromIDListW(pidl, addr path[0]) != 0: result = fromWide(addr path[0])
      coTaskMemFree(pidl)
else:
  proc dialogOutput(exe: string, args: seq[string]): string =
    try:
      let (code, text) = runBlocking(exe, args)
      if code == 0: result = text.strip(chars = {'\n', '\r'})
    except CatchableError: discard

  proc pickFile(initialDir: string): string =
    let start = if initialDir.len > 0: initialDir / "" else: ""
    if findExe("zenity").len > 0:
      result = dialogOutput(findExe("zenity"), @["--file-selection", "--title=Open an image", "--filename=" & start])
    elif findExe("kdialog").len > 0:
      result = dialogOutput(findExe("kdialog"), @["--getopenfilename", (if start.len > 0: start else: getHomeDir()), "Images (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.jxl *.avif *.heic *.bmp *.gif *.exr)"])

  proc pickFolder(initialDir: string): string =
    let start = if initialDir.len > 0: initialDir / "" else: ""
    if findExe("zenity").len > 0:
      result = dialogOutput(findExe("zenity"), @["--file-selection", "--directory", "--title=Choose the save folder", "--filename=" & start])
    elif findExe("kdialog").len > 0:
      result = dialogOutput(findExe("kdialog"), @["--getexistingdirectory", (if start.len > 0: start else: getHomeDir())])

# ---------------------------------------------------------------------------------------------
# The application: state, and what each key does
# ---------------------------------------------------------------------------------------------
type
  MenuKind = enum mkNone, mkResolution, mkScale, mkSampling
  EditField = enum efNone, efWidth, efHeight

  ExportJob = object
    pixels: Bytes
    srcW, srcH, bands, x0, y0, outW, outH, n: int
    modeCount: int                   ## how many images: one, or one per mode for "all"
    modes: array[10, int]            ## SampleMode ordinals
    paths: array[10, array[1024, char]]
    rowsDone: Atomic[int]            ## rows done over ALL the images
    cancel: Atomic[bool]
    state: Atomic[int]               ## 0 running, 1 saved, 2 failed
    message: array[300, char]

  Exporter = ref object
    job: ptr ExportJob
    thread: Thread[ptr ExportJob]
    outPath, summary: string         ## the first (or only) file, and what is being made
    count: int
    started: MonoTime

  App = ref object
    img: Image
    prep: Prep
    prepName: string
    pyr: Pyramid
    exporter: Exporter
    resIdx, customW, customH, n: int
    mode: SampleMode
    rx, ry: int                      ## top-left of the source rectangle, in source pixels
    exact, uiVisible, fullscreen, quit, dirty: bool
    menu: MenuKind
    menuSel, menuTop: int
    edit: EditField
    editBuf: string
    editFresh: bool                  ## the shown number is untouched: the first digit typed replaces it
    status: string
    statusUntil: MonoTime
    saveDir: string
    held, heldPlain: array[4, bool]  ## up, down, left, right
    heldSince: array[4, MonoTime]
    heldAcc: array[4, float]
    lastTick: MonoTime
    cols: seq[int32]
    wantDialog: int                  ## 1 = open an image, 2 = choose the save folder (done by the main loop)
    titleChanged, interactive: bool

const
  KeyUp = 1073741906
  KeyDown = 1073741905
  KeyLeft = 1073741904
  KeyRight = 1073741903
  KeyPageUp = 1073741899
  KeyPageDown = 1073741902
  KeyHome = 1073741898
  KeyEnd = 1073741901
  KeyF11 = 1073741892
  KeyKpEnter = 1073741912
  KeyEnter = 13
  KeyEsc = 27
  KeyTab = 9
  KeyBackspace = 8
  KeySpace = 32
  ModShift = 0x3
  ModCtrl = 0xC0
  MaxOutputPixels = 400_000_000

proc newApp(): App =
  result = App(resIdx: 3, customW: 1920, customH: 1080, n: 1, uiVisible: true, saveDir: LastSaveDir,
               dirty: true, lastTick: getMonoTime())

proc outRes(app: App): (int, int) =
  if app.resIdx == CustomIndex: (app.customW, app.customH)
  else: (Resolutions[app.resIdx].w, Resolutions[app.resIdx].h)

proc rectSize(app: App): (int, int) =
  let (ow, oh) = app.outRes
  (ow * app.n, oh * app.n)

proc fits(app: App, outW, outH, n: int): bool =
  app.img == nil or (outW * n <= app.img.w and outH * n <= app.img.h)

proc say(app: App, msg: string, seconds = 7.0) =
  app.status = msg
  app.statusUntil = getMonoTime() + initDuration(milliseconds = int(seconds * 1000))
  app.dirty = true

proc clampRect(app: App) =
  if app.img == nil: return
  let (rw, rh) = app.rectSize
  app.rx = clampInt(app.rx, 0, max(0, app.img.w - rw))
  app.ry = clampInt(app.ry, 0, max(0, app.img.h - rh))

proc centerRect(app: App) =
  if app.img == nil: return
  let (rw, rh) = app.rectSize
  app.rx = (app.img.w - rw) div 2
  app.ry = (app.img.h - rh) div 2
  app.clampRect()

proc changeOutput(app: App, resIdx, customW, customH, n: int, mode: SampleMode) =
  ## Change what is exported, keeping the centre of the rectangle where it is.
  let (oldW, oldH) = app.rectSize
  let cx = app.rx + oldW div 2
  let cy = app.ry + oldH div 2
  app.resIdx = resIdx
  app.customW = customW
  app.customH = customH
  app.n = n
  app.mode = mode
  let (rw, rh) = app.rectSize
  app.rx = cx - rw div 2
  app.ry = cy - rh div 2
  app.clampRect()
  app.dirty = true

proc fitToImage(app: App): bool =
  ## After an image is opened: shrink n, or pick a smaller resolution, until the rectangle fits.
  if app.img == nil: return false
  let img = app.img
  let (oldIdx, oldN) = (app.resIdx, app.n)
  let (ow, oh) = app.outRes
  var n = app.n
  while n > 1 and (ow * n > img.w or oh * n > img.h): dec n
  var idx = app.resIdx
  var cw = app.customW
  var ch = app.customH
  if ow * n > img.w or oh * n > img.h:
    var best = -1
    var bestArea = 0
    for i, r in Resolutions:
      if r.w <= img.w and r.h <= img.h and r.w * r.h > bestArea:
        best = i
        bestArea = r.w * r.h
    if best >= 0: idx = best
    else:
      idx = CustomIndex
      cw = img.w
      ch = img.h
  result = idx != oldIdx or n != oldN
  app.changeOutput(idx, cw, ch, n, app.mode)

proc refreshLevels(app: App) =
  ## Map the pyramid levels the background thread has finished since last time.
  if app.img == nil or app.pyr == nil: return
  for level in 1 ..< app.pyr.job.ready.load():
    if not mapLevel(app.img, level): break
  if app.pyr.job.finished.load():
    stopPyramid(app.pyr)
    app.pyr = nil
    app.dirty = true

proc closeCurrentImage(app: App) =
  stopPyramid(app.pyr)
  app.pyr = nil
  closeImage(app.img)
  app.img = nil

proc finishOpen(app: App, img: Image) =
  closeCurrentImage(app)
  app.img = img
  app.menu = mkNone
  app.edit = efNone
  let changed = app.fitToImage()
  app.centerRect()
  app.pyr = startPyramid(img)
  app.titleChanged = true
  app.dirty = true
  let name = splitFile(img.srcPath).name & splitFile(img.srcPath).ext
  app.say("Opened " & name & " (" & $img.w & " x " & $img.h & ")" &
          (if changed: ". The output size was reduced to fit this image." else: ""))
  if app.interactive:
    let problem = rememberInSource("LastImage", img.srcPath)
    if problem.len > 0: echo "note: ", problem

proc beginOpen(app: App, path: string) =
  if app.exporter != nil:
    app.say("Wait for the export to finish before opening another image.")
    return
  if not fileExists(path):
    app.say("There is no file " & path)
    return
  cancelPrep(app.prep)
  app.prep = nil
  let cached = openCached(path, cacheDirFor(path))
  if cached != nil:
    app.finishOpen(cached)
    return
  let (info, err) = probeImage(path)
  if err.len > 0:
    app.say(err, 12)
    return
  app.prep = startPrep(path, info)
  app.prepName = splitFile(path).name & splitFile(path).ext
  if app.prep.state == psFailed:
    app.say(app.prep.error, 12)
    app.prep = nil
  app.dirty = true

proc effectiveSaveDir(app: App): string =
  if app.saveDir.len > 0 and dirExists(app.saveDir): return app.saveDir
  if app.img != nil: return parentDir(app.img.srcPath)
  getCurrentDir()

proc exportMain(job: ptr ExportJob) {.thread.} =
  var err = ""
  for i in 0 ..< job.modeCount:           # one image, or one per mode
    var outPath = ""
    for c in job.paths[i]:
      if c == '\0': break
      outPath.add c
    err = exportPixelsToPng(job.pixels, job.srcW, job.srcH, job.bands, job.x0, job.y0, job.outW,
                            job.outH, job.n, SampleMode(job.modes[i]), outPath, addr job.rowsDone, addr job.cancel)
    if err.len > 0: break
  for i in 0 ..< min(err.len, job.message.len - 1): job.message[i] = err[i]
  job.state.store(if err.len == 0: 1 else: 2)

proc startExport(app: App) =
  if app.img == nil:
    app.say("Open an image first (O).")
    return
  if app.exporter != nil:
    app.say("An export is already running.")
    return
  let (ow, oh) = app.outRes
  if ow * oh > MaxOutputPixels:
    app.say("That output size is too large to export.")
    return
  let dir = app.effectiveSaveDir()
  try: createDir(dir)
  except CatchableError:
    app.say("Cannot create the save folder " & dir)
    return
  # what to make: one image, or - for "all" - one per mode (at n = 1 every mode gives the same pixels: one image)
  var modes: seq[SampleMode]
  if app.n == 1: modes = @[smMean]
  elif app.mode == smAll: modes = @SingleModes
  else: modes = @[app.mode]
  var paths: seq[string]
  for m in modes:
    let path = uniquePath(dir, exportName(splitFile(app.img.srcPath).name, app.rx, app.ry, ow, oh, app.n, m))
    if path.len >= 1000:
      app.say("The save path is too long.")
      return
    paths.add path
  let ex = Exporter(outPath: paths[0], count: modes.len, started: getMonoTime())
  ex.job = cast[ptr ExportJob](allocShared0(sizeof(ExportJob)))
  let j = ex.job
  j.pixels = pixelsOf(app.img, 0)
  j.srcW = app.img.w
  j.srcH = app.img.h
  j.bands = app.img.bands
  j.x0 = app.rx
  j.y0 = app.ry
  j.outW = ow
  j.outH = oh
  j.n = app.n
  j.modeCount = modes.len
  for i, m in modes:
    j.modes[i] = ord(m)
    for k in 0 ..< paths[i].len: j.paths[i][k] = paths[i][k]
  let blocks = $app.n & " x " & $app.n & " blocks"
  ex.summary = $ow & " x " & $oh &
    (if app.n == 1: " (pixel-perfect" & (if app.mode == smAll: ": every mode gives the same pixels, so one image)" else: ")")
     elif app.mode == smAll: " from " & blocks & ", one image per sampling mode"
     else: " from " & blocks & " by " & ModeLabels[app.mode])
  createThread(ex.thread, exportMain, j)
  app.exporter = ex
  app.say("Exporting " & ex.summary & " ...", 60)

proc pollWork(app: App) =
  ## Called every loop: check the conversion, the pyramid and the export.
  if app.prep != nil and app.exporter == nil:      # never swap the image while an export is reading it
    pollPrep(app.prep)
    if app.prep.state == psDone:
      let img = app.prep.image
      app.prep = nil
      app.finishOpen(img)
    elif app.prep.state == psFailed:
      app.say("Could not open the image: " & app.prep.error, 12)
      app.prep = nil
    app.dirty = true
  refreshLevels(app)
  if app.exporter != nil and app.exporter.job.state.load() != 0:
    let ex = app.exporter
    joinThread(ex.thread)
    var msg = ""
    for c in ex.job.message:
      if c == '\0': break
      msg.add c
    if ex.job.state.load() == 1:
      if ex.count > 1: app.say("Saved " & $ex.count & " images, " & ex.summary & ", in " & parentDir(ex.outPath), 20)
      else: app.say("Saved " & ex.summary & ": " & ex.outPath, 20)
    else:
      app.say("Export failed: " & msg, 20)
    deallocShared(ex.job)
    app.exporter = nil
    app.dirty = true

proc chooseSaveFolder(app: App, dir: string) =
  if dir.len == 0: return
  app.saveDir = dir
  let problem = if app.interactive: rememberInSource("LastSaveDir", dir) else: ""
  app.say("Export folder: " & dir & (if problem.len > 0: "  (not remembered: " & problem & ")" else: ""))

# --- what the menus list --------------------------------------------------------------------
proc menuItems(app: App): seq[string] =
  case app.menu
  of mkResolution:
    for r in Resolutions:
      result.add $r.w & " x " & $r.h & "  " & r.name & (if app.fits(r.w, r.h, app.n): "" else: "  - too big for this image")
    result.add "Custom ..."
  of mkScale:
    for k in 1 .. MaxScale:
      let (ow, oh) = app.outRes
      result.add $k & " x " & $k & " -> 1 x 1" & (if k == 1: "  pixel-perfect: no resampling" else: "  pixel-group-perfect") &
        (if app.fits(ow, oh, k): "" else: "  - too big for this image")
  of mkSampling:
    for m in SampleMode: result.add ModeLabels[m] & "  " & ModeNotes[m]
  of mkNone: discard

proc menuCurrent(app: App): int =
  case app.menu
  of mkResolution: app.resIdx
  of mkScale: app.n - 1
  of mkSampling: ord(app.mode)
  of mkNone: 0

proc openMenu(app: App, kind: MenuKind) =
  app.menu = kind
  app.menuSel = app.menuCurrent()
  app.menuTop = 0
  app.dirty = true

proc applyMenu(app: App) =
  let idx = app.menuSel
  case app.menu
  of mkResolution:
    if idx == CustomIndex:
      let (ow, oh) = app.outRes
      app.changeOutput(CustomIndex, ow, oh, app.n, app.mode)
      app.say("Custom size: press W to type the width and H to type the height.")
    else:
      let r = Resolutions[idx]
      if not app.fits(r.w, r.h, app.n):
        app.say("A " & $r.w & " x " & $r.h & " output at " & $app.n & " x " & $app.n & " needs more pixels than this image has.")
        return
      app.changeOutput(idx, app.customW, app.customH, app.n, app.mode)
  of mkScale:
    let (ow, oh) = app.outRes
    if not app.fits(ow, oh, idx + 1):
      app.say("A " & $(idx + 1) & " x " & $(idx + 1) & " scale of " & $ow & " x " & $oh & " needs more pixels than this image has.")
      return
    app.changeOutput(app.resIdx, app.customW, app.customH, idx + 1, app.mode)
  of mkSampling:
    app.changeOutput(app.resIdx, app.customW, app.customH, app.n, SampleMode(idx))
  of mkNone: discard
  app.menu = mkNone
  app.dirty = true

proc nudge(app: App, dir, step: int) =
  ## dir: 0 up, 1 down, 2 left, 3 right
  if app.img == nil: return
  case dir
  of 0: app.ry -= step
  of 1: app.ry += step
  of 2: app.rx -= step
  else: app.rx += step
  app.clampRect()
  app.dirty = true

proc startEdit(app: App, field: EditField) =
  if app.resIdx != CustomIndex:
    app.say("Choose \"Custom\" in the output size list first (R).")
    return
  app.edit = field
  app.editBuf = $(if field == efWidth: app.customW else: app.customH)
  app.editFresh = true
  app.dirty = true

proc commitEdit(app: App): bool =
  ## Apply the typed number.  False (and a message) when it is not usable.
  var v = 0
  try: v = parseInt(app.editBuf)
  except ValueError:
    app.say("Type a whole number.")
    return false
  let (rw, rh) = (app.customW, app.customH)
  let (newW, newH) = if app.edit == efWidth: (v, rh) else: (rw, v)
  if v < 1 or newW * newH > MaxOutputPixels:
    app.say("That size is not usable.")
    return false
  if not app.fits(newW, newH, app.n):
    app.say($newW & " x " & $newH & " at " & $app.n & " x " & $app.n & " needs more pixels than this image has.")
    return false
  app.changeOutput(CustomIndex, newW, newH, app.n, app.mode)
  true

proc handleText(app: App, text: string) =
  if app.edit == efNone: return
  for c in text:
    if c in {'0' .. '9'}:
      if app.editFresh:
        app.editBuf = ""
        app.editFresh = false
      if app.editBuf.len < 6: app.editBuf.add c
  app.dirty = true

proc handleKey(app: App, sym, mods: int, repeat: bool) =
  let ctrl = (mods and ModCtrl) != 0
  let shift = (mods and ModShift) != 0
  app.dirty = true
  if ctrl and sym == ord('q'):
    app.quit = true
    return
  # the operating system repeats a held key; only moving (arrows, list navigation, Backspace) should repeat
  let movesOrDeletes = sym in [KeyUp, KeyDown, KeyLeft, KeyRight, KeyPageUp, KeyPageDown, KeyHome, KeyEnd, KeyBackspace]
  if repeat and not movesOrDeletes: return
  # --- typing a custom width or height
  if app.edit != efNone:
    case sym
    of KeyEnter, KeyKpEnter:
      if app.commitEdit(): app.edit = efNone
    of KeyEsc: app.edit = efNone
    of KeyBackspace:
      if app.editFresh:
        app.editBuf = ""
        app.editFresh = false
      elif app.editBuf.len > 0: app.editBuf.setLen(app.editBuf.len - 1)
    of KeyTab:
      if app.commitEdit(): app.startEdit(if app.edit == efWidth: efHeight else: efWidth)
    else: discard
    return
  # --- a drop-down list is open
  if app.menu != mkNone:
    let items = app.menuItems()
    case sym
    of KeyUp: app.menuSel = max(0, app.menuSel - 1)
    of KeyDown: app.menuSel = min(items.len - 1, app.menuSel + 1)
    of KeyPageUp: app.menuSel = max(0, app.menuSel - 8)
    of KeyPageDown: app.menuSel = min(items.len - 1, app.menuSel + 8)
    of KeyHome: app.menuSel = 0
    of KeyEnd: app.menuSel = items.len - 1
    of KeyEnter, KeyKpEnter, KeySpace: app.applyMenu()
    of KeyEsc: app.menu = mkNone
    of ord('r'):
      if app.menu == mkResolution: app.menu = mkNone
      else: app.openMenu(mkResolution)
    of ord('s'):
      if app.menu == mkScale: app.menu = mkNone
      else: app.openMenu(mkScale)
    of ord('m'):
      if app.menu == mkSampling: app.menu = mkNone
      else: app.openMenu(mkSampling)
    of ord('1') .. ord('9'):
      if app.menu == mkScale and sym - ord('1') < items.len:
        app.menuSel = sym - ord('1')
        app.applyMenu()
    else: discard
    return
  # --- the usual keys
  case sym
  of ord('r'): app.openMenu(mkResolution)
  of ord('s'): app.openMenu(mkScale)
  of ord('m'): app.openMenu(mkSampling)
  of ord('o'): app.wantDialog = 1
  of ord('f'): app.wantDialog = 2
  of ord('e'): app.startExport()
  of ord('z'): app.exact = not app.exact
  of ord('c'): app.centerRect()
  of ord('w'): app.startEdit(efWidth)
  of ord('h'): app.startEdit(efHeight)
  of KeyTab: app.uiVisible = not app.uiVisible
  of KeyF11:
    app.fullscreen = not app.fullscreen
  of KeyUp, KeyDown, KeyLeft, KeyRight:
    let d = (case sym
             of KeyUp: 0
             of KeyDown: 1
             of KeyLeft: 2
             else: 3)
    let (rw, rh) = app.rectSize
    if ctrl:
      app.nudge(d, app.n)                                       # one output pixel
    elif shift:
      app.nudge(d, max(1, (if d < 2: rh else: rw) div 4))       # a quarter of the rectangle
    elif not repeat:
      app.nudge(d, 1)                                           # exactly one source pixel; holding speeds up
      app.held[d] = true
      app.heldPlain[d] = true
      app.heldSince[d] = getMonoTime()
      app.heldAcc[d] = 0
  else: discard

proc handleKeyUp(app: App, sym: int) =
  case sym
  of KeyUp: app.held[0] = false
  of KeyDown: app.held[1] = false
  of KeyLeft: app.held[2] = false
  of KeyRight: app.held[3] = false
  else: discard

proc updateHeld(app: App, now: MonoTime) =
  ## Holding an arrow key: after a short delay the rectangle moves faster and faster.
  let dt = float((now - app.lastTick).inMicroseconds) / 1e6
  app.lastTick = now
  for d in 0 .. 3:
    if not app.held[d] or not app.heldPlain[d]: continue
    let t = float((now - app.heldSince[d]).inMicroseconds) / 1e6
    if t < 0.35: continue
    let (rw, rh) = app.rectSize
    let top = float(max(rw, rh)) * 3.0
    let speed = min(top, HoldStartSpeed * pow(2.0, (t - 0.35) / 0.35))
    app.heldAcc[d] += speed * min(dt, 0.1)
    let steps = int(app.heldAcc[d])
    if steps > 0:
      app.heldAcc[d] -= float(steps)
      app.nudge(d, steps)

proc isBusy(app: App): bool =
  app.prep != nil or app.exporter != nil or app.pyr != nil

# ---------------------------------------------------------------------------------------------
# Drawing a frame
# ---------------------------------------------------------------------------------------------
proc drawPicture(app: App, fb: var Fb, cam: Cam) =
  const background = 0xFF161616'u32
  for i in 0 ..< fb.px.len: fb.px[i] = background
  if app.img == nil: return
  let img = app.img
  var level = 0
  if cam.s < 1.0:
    level = clampInt(int(floor(log2(1.0 / cam.s) + 0.5)), 0, img.levelCount - 1)
  while level > 0 and not (level < img.levels.len and img.levels[level].mapped): dec level
  let lw = img.levels[level].w
  let lh = img.levels[level].h
  let src = pixelsOf(img, level)
  let bands = img.bands
  let unit = float(1 shl level)
  if app.cols.len != fb.w: app.cols = newSeq[int32](fb.w)
  for x in 0 ..< fb.w:
    let sx = float(app.rx) + (float(x) + 0.5 - float(cam.left)) / cam.s
    app.cols[x] = if sx < 0 or sx >= float(img.w): -1'i32 else: int32(min(int(sx / unit), lw - 1))
  let dst = cast[Pixels](addr fb.px[0])
  var prevIy = -1
  var prevY = 0
  for y in 0 ..< fb.h:
    let sy = float(app.ry) + (float(y) + 0.5 - float(cam.top)) / cam.s
    if sy < 0 or sy >= float(img.h):
      prevIy = -1
      continue
    let iy = min(int(sy / unit), lh - 1)
    let row = y * fb.w
    if iy == prevIy:
      copyMem(addr dst[row], addr dst[prevY * fb.w], fb.w * 4)
    else:
      let line = src.byteAt(iy * lw * bands)
      for x in 0 ..< fb.w:
        let c = app.cols[x]
        if c >= 0:
          let o = int(c) * bands
          dst[row + x] = 0xFF000000'u32 or (uint32(line[o]) shl 16) or (uint32(line[o + 1]) shl 8) or uint32(line[o + 2])
      prevIy = iy
      prevY = y

proc outlinePixel(fb: var Fb, x, y: int) =
  if x < 0 or y < 0 or x >= fb.w or y >= fb.h: return
  let p = fb.px[y * fb.w + x]
  fb.px[y * fb.w + x] = if outlineIsWhite(int((p shr 16) and 255), int((p shr 8) and 255), int(p and 255)): White else: Black

proc drawOutline(fb: var Fb, cam: Cam) =
  ## One pixel wide, just inside the rectangle; every pixel is white or black, whichever contrasts
  ## more with the picture under it.
  let x0 = cam.left
  let y0 = cam.top
  let x1 = cam.left + cam.rwS - 1
  let y1 = cam.top + cam.rhS - 1
  # every pixel of the ring exactly once: a pixel that is drawn already is no longer the picture
  for x in max(0, x0) .. min(fb.w - 1, x1):
    fb.outlinePixel(x, y0)
    if y1 > y0: fb.outlinePixel(x, y1)
  for y in max(0, y0 + 1) .. min(fb.h - 1, y1 - 1):
    fb.outlinePixel(x0, y)
    if x1 > x0: fb.outlinePixel(x1, y)

proc fitLine(s: string, maxPx, sc: int): string =
  let maxChars = max(8, (maxPx + sc) div (6 * sc))
  truncateMiddle(s, maxChars)

proc elapsedText(since: MonoTime): string =
  let secs = int((getMonoTime() - since).inSeconds)
  $(secs div 60) & ":" & align($(secs mod 60), 2, '0')

proc mainLines(app: App): seq[string] =
  ## The most important information: source rectangle -> saved size, n x n -> 1 x 1, the sampling mode.
  let (ow, oh) = app.outRes
  let (rw, rh) = app.rectSize
  result.add "[R] " & $rw & " x " & $rh & " -> " & $ow & " x " & $oh &
    (if app.resIdx == CustomIndex: "  custom" else: "  " & Resolutions[app.resIdx].name)
  result.add "[S] " & $app.n & " x " & $app.n & " -> 1 x 1" &
    (if app.n == 1: "  pixel-perfect" else: "  pixel-group-perfect")
  result.add "[M] by " & ModeLabels[app.mode] &
    (if app.n == 1: "  (no effect at 1 x 1)" elif app.mode == smAll: "  (" & $SingleModes.len & " images)" else: "")
  if app.resIdx == CustomIndex:
    let wText = if app.edit == efWidth: app.editBuf & "_" else: $app.customW
    let hText = if app.edit == efHeight: app.editBuf & "_" else: $app.customH
    result.add "[W] width " & wText & "    [H] height " & hText
    if app.edit != efNone: result.add "Enter = apply   Tab = other field   Esc = cancel"

proc drawUi(app: App, fb: var Fb) =
  let u = max(2, fb.h div 540)          # small text
  let big = u + 1                       # the main block
  let m = 4 * u
  var mainBottom = m
  var lowTop = fb.h - m
  if app.uiVisible:
    var lines = app.mainLines()
    for i in 0 ..< lines.len: lines[i] = fitLine(lines[i], fb.w - 2 * m - 4 * big, big)
    let box = fb.drawPanel(m, m, lines, big)
    mainBottom = m + box.h
    # --- the lower block: image, folder, keys
    var low: seq[string]
    if app.img != nil:
      let name = splitFile(app.img.srcPath).name & splitFile(app.img.srcPath).ext
      low.add "[O] open image   " & name & "  (" & $app.img.w & " x " & $app.img.h & ", " &
        (if app.img.bands == 4: "with alpha" else: "RGB") & ")"
      low.add "[arrows] move the rectangle: x " & $app.rx & "  y " & $app.ry & "  (Ctrl: 1 output pixel, Shift: far)"
    else:
      low.add "[O] open image   none yet"
    low.add "[F] save folder   " & app.effectiveSaveDir()
    low.add "[E] export   [Z] exact view   [C] centre   [Tab] hide this   [F11] fullscreen"
    if app.pyr != nil:
      let total = app.img.levelCount
      low.add "building zoom levels " & $min(total, app.pyr.job.ready.load()) & " of " & $total
    for i in 0 ..< low.len: low[i] = fitLine(low[i], fb.w - 2 * m - 4 * u, u)
    let h = low.len * 9 * u + 2 * u
    lowTop = fb.h - m - h
    discard fb.drawPanel(m, lowTop, low, u)
  # --- what is going on
  var note = ""
  if app.prep != nil:
    note = "Converting " & app.prepName & " for fast viewing (once per image): " & elapsedText(app.prep.started)
  elif app.exporter != nil:
    let done = app.exporter.job.rowsDone.load()
    note = "Exporting " & app.exporter.summary & ": " & $(100 * done div max(1, app.exporter.job.outH * app.exporter.job.modeCount)) & "%"
  elif app.status.len > 0 and getMonoTime() < app.statusUntil:
    note = app.status
  if note.len > 0:
    let noteLines = @[fitLine(note, fb.w - 2 * m - 4 * u, u)]
    let noteH = 9 * u + 2 * u
    discard fb.drawPanel(m, max(mainBottom + m, lowTop - noteH - 2 * u), noteLines, u)
  if app.img == nil and app.prep == nil:
    let msg = "Press O to open an image, or drop one on this window"
    let w = textWidth(msg, big)
    discard fb.drawPanel(max(m, (fb.w - w) div 2 - 2 * big), fb.h div 2 - 4 * big, @[msg], big)
  # --- the open drop-down list
  if app.menu != mkNone:
    let items = app.menuItems()
    let noteSpace = if note.len > 0: 11 * u + 2 * u else: 0
    let avail = lowTop - noteSpace - (mainBottom + m) - m
    let maxRows = max(3, avail div (9 * big) - 3)       # three lines are for "more above", "more below" and the help
    if app.menuSel < app.menuTop: app.menuTop = app.menuSel
    if app.menuSel >= app.menuTop + maxRows: app.menuTop = app.menuSel - maxRows + 1
    var shown, tails: seq[string]
    let last = min(items.len, app.menuTop + maxRows)
    if app.menuTop > 0:
      shown.add "  ^ more above"
      tails.add ""
    let selectedLine = shown.len + app.menuSel - app.menuTop
    let room = fb.w - 2 * m - 6 * big
    for i in app.menuTop ..< last:
      let marker = if i == app.menuSel: "> " else: "  "
      if app.menu == mkSampling:
        # the name in the big size, its explanation after it in the small one
        let parts = items[i].split("  ", 1)
        let head = fitLine(parts[0], room - textWidth(marker, big), big)
        shown.add marker & head
        let left = room - textWidth(marker & head, big) - tailGap(big)
        tails.add (if parts.len > 1 and left > 6 * u: fitLine(parts[1], left, u) else: "")
      else:
        shown.add marker & fitLine(items[i], room, big)
        tails.add ""
    if last < items.len:
      shown.add "  v more below"
      tails.add ""
    shown.add "  Up/Down, Enter = choose, Esc = close"
    tails.add ""
    discard fb.drawPanel(m, mainBottom + m, shown, big, selectedLine, tails, u)

proc dimOutside(fb: var Fb, cam: Cam) =
  ## 75% opaque black over everything outside the rectangle (the margins of the exact view).
  const quarter = 0xFF000000'u32
  for y in 0 ..< fb.h:
    let rowInside = y >= cam.top and y < cam.top + cam.rhS
    for x in 0 ..< fb.w:
      if rowInside and x >= cam.left and x < cam.left + cam.rwS: continue
      let i = y * fb.w + x
      fb.px[i] = quarter or ((fb.px[i] shr 2) and 0x003F3F3F'u32)

proc renderFrame(app: App, fb: var Fb) =
  let (rw, rh) = app.rectSize
  let cam = makeCam(fb.w, fb.h, rw, rh, app.exact)
  app.drawPicture(fb, cam)
  if app.exact and app.img != nil: dimOutside(fb, cam)
  if app.img != nil: drawOutline(fb, cam)
  app.drawUi(fb)

# ---------------------------------------------------------------------------------------------
# Keys given as text (for --snapshot), and the window loop
# ---------------------------------------------------------------------------------------------
proc keyFromName(tok: string): tuple[sym, mods: int, ok: bool] =
  var mods = 0
  var name = tok.toLowerAscii
  while true:
    if name.startsWith("shift+"):
      mods = mods or ModShift
      name = name[6 .. ^1]
    elif name.startsWith("ctrl+"):
      mods = mods or ModCtrl
      name = name[5 .. ^1]
    else: break
  result.mods = mods
  result.ok = true
  case name
  of "up": result.sym = KeyUp
  of "down": result.sym = KeyDown
  of "left": result.sym = KeyLeft
  of "right": result.sym = KeyRight
  of "pageup": result.sym = KeyPageUp
  of "pagedown": result.sym = KeyPageDown
  of "home": result.sym = KeyHome
  of "end": result.sym = KeyEnd
  of "enter": result.sym = KeyEnter
  of "esc": result.sym = KeyEsc
  of "tab": result.sym = KeyTab
  of "space": result.sym = KeySpace
  of "backspace": result.sym = KeyBackspace
  of "f11": result.sym = KeyF11
  else:
    if name.len == 1: result.sym = ord(name[0])
    else: result.ok = false

proc applyKeyScript(app: App, script: string) =
  ## Keys separated by spaces: r down enter  shift+right  text:1920
  for tok in script.splitWhitespace:
    if tok.startsWith("text:"):
      app.handleText(tok[5 .. ^1])
    else:
      let k = keyFromName(tok)
      if not k.ok:
        echo "unknown key in --keys: ", tok
        continue
      app.handleKey(k.sym, k.mods, false)
      app.handleKeyUp(k.sym)
      if app.wantDialog != 0: app.wantDialog = 0        # no dialogs when scripted

proc runDialogs(app: App) =
  let which = app.wantDialog
  app.wantDialog = 0
  if which == 1:
    let start = if app.img != nil: parentDir(app.img.srcPath) elif LastImage.len > 0: parentDir(LastImage) else: ""
    let path = pickFile(start)
    if path.len > 0: app.beginOpen(path)
  elif which == 2:
    let dir = pickFolder(app.effectiveSaveDir())
    if dir.len > 0: app.chooseSaveFolder(dir)
  app.dirty = true

proc runWindow(startPath: string): int =
  let err = loadSdl()
  if err.len > 0:
    echo err
    return 1
  makeDpiAware()
  discard sdlSetHint("SDL_RENDER_SCALE_QUALITY", "0")
  discard sdlSetHint("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
  if sdlInit(SdlInitVideo) != 0:
    echo "SDL_Init failed: ", sdlGetError()
    return 1
  var bounds: SdlRect
  var winW = 1280
  var winH = 720
  if sdlGetDisplayBounds(0, addr bounds) == 0:
    winW = max(640, bounds.w * 8 div 10)
    winH = max(400, bounds.h * 8 div 10)
  let win = sdlCreateWindow("wallcrop", SdlWindowPosCentered, SdlWindowPosCentered, winW.cint, winH.cint,
                            SdlWindowShown or SdlWindowResizable or SdlWindowHighDpi)
  if win == nil:
    echo "cannot create the window: ", sdlGetError()
    return 1
  sdlSetWindowMinimumSize(win, 480, 300)
  var ren = sdlCreateRenderer(win, -1, SdlRendererAccelerated or SdlRendererVsync)
  if ren == nil: ren = sdlCreateRenderer(win, -1, 0)
  if ren == nil:
    echo "cannot create the renderer: ", sdlGetError()
    return 1
  var tex: pointer = nil
  var texW, texH = 0
  var fb: Fb
  let app = newApp()
  app.interactive = true
  initLuts()
  initFont()
  var first = startPath
  if first.len == 0 and LastImage.len > 0 and fileExists(LastImage): first = LastImage
  if first.len > 0: app.beginOpen(first)
  var textOn = false
  var wasFullscreen = false
  while not app.quit:
    # --- wait for something to happen
    var timeout = 400
    if app.isBusy(): timeout = 100
    for d in 0 .. 3:
      if app.held[d]: timeout = 8
    var ev: SdlEvent
    var got = sdlWaitEventTimeout(addr ev, timeout.cint)
    while got != 0:
      let kind = u32At(ev, 0)
      if kind == EvQuit: app.quit = true
      elif kind == EvWindow:
        app.dirty = true
        if u8At(ev, 12) == WinFocusLost: app.held = [false, false, false, false]
      elif kind == EvKeyDown:
        app.handleKey(int(i32At(ev, 20)), int(u16At(ev, 24)), u8At(ev, 13) != 0)
      elif kind == EvKeyUp:
        app.handleKeyUp(int(i32At(ev, 20)))
      elif kind == EvTextInput:
        app.handleText(textAt(ev, 12))
      elif kind == EvDropFile:
        let p = ptrAt(ev, 8)
        if p != nil:
          let path = $cast[cstring](p)
          sdlFree(p)
          app.beginOpen(path)
      got = sdlPollEvent(addr ev)
    let now = getMonoTime()
    app.updateHeld(now)
    app.pollWork()
    if app.status.len > 0 and app.statusUntil < now and app.statusUntil != default(MonoTime):
      app.status = ""
      app.statusUntil = default(MonoTime)
      app.dirty = true
    # --- the window follows the state
    if app.fullscreen != wasFullscreen:
      discard sdlSetWindowFullscreen(win, if app.fullscreen: SdlWindowFullscreenDesktop else: 0'u32)
      wasFullscreen = app.fullscreen
      app.dirty = true
    if app.titleChanged:
      app.titleChanged = false
      if app.img != nil: sdlSetWindowTitle(win, cstring("wallcrop - " & splitFile(app.img.srcPath).name))
    let wantText = app.edit != efNone
    if wantText != textOn:
      textOn = wantText
      if textOn: sdlStartTextInput() else: sdlStopTextInput()
    # --- draw
    if app.dirty or app.isBusy():
      var ow, oh: cint
      discard sdlGetRendererOutputSize(ren, addr ow, addr oh)
      if ow >= 64 and oh >= 64:
        if tex == nil or texW != ow or texH != oh:
          if tex != nil: sdlDestroyTexture(tex)
          tex = sdlCreateTexture(ren, SdlPixelFormatArgb8888, SdlTextureStreaming.cint, ow, oh)
          texW = ow
          texH = oh
        fb.resizeFb(ow, oh)
        app.renderFrame(fb)
        discard sdlUpdateTexture(tex, nil, addr fb.px[0], (ow * 4).cint)
        discard sdlRenderCopy(ren, tex, nil, nil)
        sdlRenderPresent(ren)
      app.dirty = false
    if app.wantDialog != 0: app.runDialogs()
  # --- shut down: stop the background work first
  if app.exporter != nil:
    app.exporter.job.cancel.store(true)
    joinThread(app.exporter.thread)
    deallocShared(app.exporter.job)
  cancelPrep(app.prep)
  app.closeCurrentImage()
  if tex != nil: sdlDestroyTexture(tex)
  sdlDestroyRenderer(ren)
  sdlDestroyWindow(win)
  sdlQuit()
  0


# ---------------------------------------------------------------------------------------------
# Self-test and command-line modes
# ---------------------------------------------------------------------------------------------
var testsRun, testsFailed = 0

proc check(cond: bool, what: string) =
  inc testsRun
  if not cond:
    inc testsFailed
    echo "  FAIL  ", what

proc refBlock(src: seq[uint8], srcW, bands, sx, sy, n: int, mode: SampleMode): seq[int] =
  ## Slow, plain reference for ONE output pixel: the block as a list of pixels, then what the mode says.
  var px: seq[seq[int]]
  for dy in 0 ..< n:
    for dx in 0 ..< n:
      var p: seq[int]
      for ch in 0 ..< bands: p.add int(src[((sy + dy) * srcW + sx + dx) * bands + ch])
      px.add p
  let nn = px.len
  let colours = min(bands, 3)
  proc values(ch: int): seq[int] =
    for p in px: result.add p[ch]
  proc twiceMedian(ch: int): int =
    var v = values(ch)
    v.sort()
    if nn mod 2 == 1: 2 * v[nn div 2] else: v[nn div 2 - 1] + v[nn div 2]
  var sums = newSeq[int](bands)
  for ch in 0 ..< bands:
    for v in values(ch): sums[ch] += v
  result = newSeq[int](bands)
  case mode
  of smMean, smAll:
    for ch in 0 ..< bands: result[ch] = (sums[ch] * 2 + nn) div (2 * nn)          # round half up
  of smMedian:
    for ch in 0 ..< bands: result[ch] = (twiceMedian(ch) + 1) div 2
  of smMax:
    for ch in 0 ..< bands: result[ch] = max(values(ch))
  of smMin:
    for ch in 0 ..< bands: result[ch] = min(values(ch))
  of smTopLeft:
    result = px[0]
  of smRandomColor:
    result = px[pickIndex(sx, sy, 0'u64, nn)]
  of smRandomChannels:
    for ch in 0 ..< bands: result[ch] = px[pickIndex(sx, sy, channelSalt(ch), nn)][ch]
  of smFarthestMeanColor:
    var best = 0
    var bestD = -1
    for k in 0 ..< nn:
      var d = 0
      for ch in 0 ..< colours: d += (px[k][ch] * nn - sums[ch]) * (px[k][ch] * nn - sums[ch])
      if d > bestD:
        bestD = d
        best = k
    result = px[best]
  of smFarthestMeanChannel:
    for ch in 0 ..< bands:
      var bestD = -1
      for k in 0 ..< nn:
        if abs(px[k][ch] * nn - sums[ch]) > bestD:
          bestD = abs(px[k][ch] * nn - sums[ch])
          result[ch] = px[k][ch]
  of smFarthestMedianChannel:
    for ch in 0 ..< bands:
      let twice = twiceMedian(ch)
      var bestD = -1
      for k in 0 ..< nn:
        if abs(2 * px[k][ch] - twice) > bestD:
          bestD = abs(2 * px[k][ch] - twice)
          result[ch] = px[k][ch]

proc selfTest(): int =
  initLuts()
  randomize(12345)
  # --- selection and median against sorting
  for m in 1 .. 64:
    for round in 0 ..< 40:
      var a: array[64, uint8]
      var vals: seq[int]
      let spread = if round mod 4 == 0: 3 else: 256            # many duplicates in some rounds
      for i in 0 ..< m:
        a[i] = uint8(rand(spread - 1))
        vals.add int(a[i])
      vals.sort()
      let expected = if m mod 2 == 1: vals[m div 2] else: (vals[m div 2 - 1] + vals[m div 2] + 1) div 2
      check(medianOf(a, m) == expected, "median of " & $m & " values")
  # --- the random modes are fixed by the seed and the block's place: values from an independent Python version
  check(pickIndex(0, 0, 0'u64, 64) == 59 and pickIndex(1234, 5678, 0'u64, 9) == 0 and
        pickIndex(10, 20, channelSalt(2), 16) == 7 and pickIndex(31000, 32000, channelSalt(0), 4) == 0 and
        pickIndex(7, 3, channelSalt(3), 25) == 16, "the random pick matches the reference implementation")
  # --- a block worked out by hand, one value per mode (2 x 2 pixels, RGB)
  block:
    let src = @[10'u8, 200, 30, 250, 20, 40, 12, 18, 50, 14, 16, 60]            # four pixels, row by row
    let want: array[SampleMode, array[3, int]] = [
      [72, 64, 45], [13, 19, 45], [250, 200, 60], [10, 16, 30], [-1, -1, -1], [-1, -1, -1], [10, 200, 30],
      [250, 20, 40], [250, 200, 30], [250, 200, 30], [-1, -1, -1]]
    for m in SingleModes:
      var dst = newSeq[uint8](3)
      exportBlocks(cast[Bytes](unsafeAddr src[0]), 2, 3, 0, 0, 1, 1, 2, m, cast[Bytes](addr dst[0]))
      if want[m][0] >= 0:
        check(@[int(dst[0]), int(dst[1]), int(dst[2])] == @(want[m]), "hand-worked block, mode " & ModeLabels[m] & ": " & $dst)
      else:          # random: every channel is one of the block's values for that channel
        let r = refBlock(src, 2, 3, 0, 0, 2, m)
        check(@[int(dst[0]), int(dst[1]), int(dst[2])] == r, "random mode matches its reference: " & $dst)
        if m == smRandomColor:
          var isAPixel = false
          for k in 0 ..< 4:
            if int(dst[0]) == int(src[k * 3]) and int(dst[1]) == int(src[k * 3 + 1]) and int(dst[2]) == int(src[k * 3 + 2]): isAPixel = true
          check(isAPixel, "colorwise random sample is one whole pixel of the block")
  # --- export: every mode of n x n blocks against the slow reference, 3 and 4 channels
  for bands in [3, 4]:
    let srcW = 41
    let srcH = 37
    var src = newSeq[uint8](srcW * srcH * bands)
    for i in 0 ..< src.len: src[i] = uint8(rand(255))
    for n in 1 .. MaxScale:
      for mode in SingleModes:
        let outW = min(5, (srcW - 3) div n)
        let outH = min(4, (srcH - 2) div n)
        if outW < 1 or outH < 1: continue
        let x0 = 3
        let y0 = 2
        var dst = newSeq[uint8](outW * outH * bands)
        exportBlocks(cast[Bytes](addr src[0]), srcW, bands, x0, y0, outW, outH, n, mode, cast[Bytes](addr dst[0]))
        var bad = 0
        for oy in 0 ..< outH:
          for ox in 0 ..< outW:
            let want = refBlock(src, srcW, bands, x0 + ox * n, y0 + oy * n, n, mode)
            for ch in 0 ..< bands:
              if int(dst[(oy * outW + ox) * bands + ch]) != want[ch]: inc bad
        check(bad == 0, "export n=" & $n & " bands=" & $bands & " mode=" & ModeLabels[mode] & " (" & $bad & " wrong)")
  # a flat block stays flat, and extreme values do not overflow
  block:
    var src = newSeq[uint8](8 * 8 * 3)
    for i in 0 ..< src.len: src[i] = 255
    var dst = newSeq[uint8](3)
    exportBlocks(cast[Bytes](addr src[0]), 8, 3, 0, 0, 1, 1, 8, smMean, cast[Bytes](addr dst[0]))
    check(dst[0] == 255 and dst[1] == 255 and dst[2] == 255, "mean of 64 x 255")
    for m in SingleModes:
      exportBlocks(cast[Bytes](addr src[0]), 8, 3, 0, 0, 1, 1, 8, m, cast[Bytes](addr dst[0]))
      check(dst[0] == 255 and dst[1] == 255 and dst[2] == 255, "every mode of a flat block of 64 x 255: " & ModeLabels[m])
  # --- pyramid levels
  check(levelSize(32768, 32768, 3) == (4096, 4096), "level size 3 of 32768")
  check(levelSize(101, 51, 2) == (26, 13), "level size rounds up")
  check(levelCountFor(512, 512) == 1 and levelCountFor(513, 100) == 2 and levelCountFor(32768, 32768) == 7,
        "level counts")
  block:
    let sw = 7
    let sh = 5
    let bands = 3
    var src = newSeq[uint8](sw * sh * bands)
    for i in 0 ..< src.len: src[i] = uint8(rand(255))
    let dw = halfCeil(sw)
    let dh = halfCeil(sh)
    var dst = newSeq[uint8](dw * dh * bands)
    var ctx = LevelCtx(src: cast[Bytes](addr src[0]), dst: cast[Bytes](addr dst[0]), sw: sw, sh: sh,
                       dw: dw, dh: dh, bands: bands)
    levelRows(addr ctx, 0, dh)
    var bad = 0
    for y in 0 ..< dh:
      for x in 0 ..< dw:
        for ch in 0 ..< bands:
          var s = 0
          for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            s += int(src[(min(2 * y + dy, sh - 1) * sw + min(2 * x + dx, sw - 1)) * bands + ch])
          if int(dst[(y * dw + x) * bands + ch]) != (s + 2) div 4: inc bad
    check(bad == 0, "pyramid level is the 2 x 2 mean with repeated edges")
  # --- camera
  block:
    let c = makeCam(1920, 1080, 1920, 1080, true)
    check(c.s == 1.0 and c.left == 0 and c.top == 0 and c.rwS == 1920 and c.rhS == 1080, "exact camera 1:1")
    let c2 = makeCam(1920, 1080, 3840, 2160, true)
    check(c2.s == 0.5 and c2.left == 0 and c2.rwS == 1920, "exact camera of a 2 x 2 rectangle")
    let c3 = makeCam(1920, 1080, 1920, 1080, false)
    check(abs(c3.s - 0.8) < 1e-12 and c3.left == 192 and c3.top == 108 and c3.rwS == 1536, "normal camera is zoomed out")
    let c4 = makeCam(1000, 1000, 1920, 1080, true)
    check(c4.left == 0 and c4.rwS == 1000 and c4.top == 219 and c4.rhS == 563, "exact camera letterboxes")
  # --- outline colour
  check(outlineIsWhite(0, 0, 0) and not outlineIsWhite(255, 255, 255), "outline: white on black, black on white")
  block:
    var firstBlackGrey = -1
    for v in 0 .. 255:
      if not outlineIsWhite(v, v, v):
        firstBlackGrey = v
        break
    check(firstBlackGrey in 100 .. 130, "outline switches to black at a middle grey (" & $firstBlackGrey & ")")
    for v in 0 .. 255:    # contrast ratios agree with the choice
      let lum = float(srgbLinear[v])
      let withWhite = 1.05 / (lum + 0.05)
      let withBlack = (lum + 0.05) / 0.05
      check(outlineIsWhite(v, v, v) == (withWhite > withBlack), "outline contrast choice for grey " & $v)
  # --- remembering paths in the source text
  block:
    let crlf = "a\r\nconst LastImage = \"\"\r\nconst LastSaveDir = \"x\"\r\nz\r\n"
    let (t, found) = replaceConstLine(crlf, "LastImage", r"C:\pics\a ""b"".png")
    check(found and t == "a\r\nconst LastImage = \"C:\\\\pics\\\\a \\\"b\\\".png\"\r\nconst LastSaveDir = \"x\"\r\nz\r\n",
          "replaceConstLine keeps CRLF and escapes")
    let lf = "const LastImage = \"\"\nconst LastSaveDir = \"x\"\n"
    let (t2, f2) = replaceConstLine(lf, "LastSaveDir", "caf\xC3\xA9")
    check(f2 and t2 == "const LastImage = \"\"\nconst LastSaveDir = \"caf\\xC3\\xA9\"\n", "replaceConstLine escapes non-ASCII bytes")
    check(not replaceConstLine("nothing\n", "LastImage", "x").found, "replaceConstLine reports a missing line")
    let (t3, f3) = replaceConstLine("const LastImage = \"\"", "LastImage", "q")      # no final newline
    check(f3 and t3 == "const LastImage = \"q\"", "replaceConstLine without a final newline")
  # --- vipsheader output and conversion decision
  block:
    let text = "4096\n4096\n3\n((VipsBandFormat) VIPS_FORMAT_UCHAR)\n((VipsInterpretation) VIPS_INTERPRETATION_sRGB)\n"
    let (ok, info) = parseHeaderOutput(text)
    check(ok and info.w == 4096 and info.h == 4096 and info.bands == 3 and not needsConversion(info), "parse vipsheader, sRGB uchar")
    let (ok2, info2) = parseHeaderOutput("warning: x\n10\n20\n1\n((VipsBandFormat) VIPS_FORMAT_UCHAR)\n((VipsInterpretation) VIPS_INTERPRETATION_B_W)")
    check(ok2 and info2.w == 10 and info2.h == 20 and needsConversion(info2), "parse vipsheader, grey needs conversion")
    check(not parseHeaderOutput("garbage").ok, "parse vipsheader rejects garbage")
    var cmyk = ImageInfo(w: 1, h: 1, bands: 4, format: "((VipsBandFormat) VIPS_FORMAT_UCHAR)",
                         interp: "((VipsInterpretation) VIPS_INTERPRETATION_CMYK)")
    check(needsConversion(cmyk), "CMYK needs conversion")
    cmyk.interp = "((VipsInterpretation) VIPS_INTERPRETATION_sRGB)"
    cmyk.format = "((VipsBandFormat) VIPS_FORMAT_USHORT)"
    check(needsConversion(cmyk), "16-bit needs conversion")
  # --- names and paths
  check(fnv64("") == 14695981039346656037'u64 and fnv64("a") == 0xaf63dc4c8601ec8c'u64, "FNV-1a values")
  check(safeStem("a[b]/c:d.png") == "a_b__c_d.png", "safeStem")
  check(exportName("pic", 100, 200, 1920, 1080, 1, smMean) == "pic_1920x1080_at_100_200_1x1.png", "export name 1x1")
  check(exportName("pic", 0, 0, 1920, 1080, 3, smMedian) == "pic_1920x1080_at_0_0_3x3_median.png", "export name 3x3 median")
  check(exportName("pic", 5, 6, 64, 48, 2, smFarthestMedianChannel) == "pic_64x48_at_5_6_2x2_far-median-channels.png", "export name by mode")
  block:
    var names: seq[string]
    for m in SampleMode: names.add ModeFileNames[m]
    var allDifferent = true
    for i in 0 ..< names.len:
      for j in i + 1 ..< names.len:
        if names[i] == names[j]: allDifferent = false
    check(allDifferent, "every mode has its own file name")
    check(parseMode("far-mean-color") == (true, smFarthestMeanColor) and parseMode("MEDIAN") == (true, smMedian) and
          not parseMode("nonsense").ok, "parseMode")
  check(truncateMiddle("abcdefghijklmnopqrstuvwxyz", 12) == "abcde...wxyz" or truncateMiddle("abcdefghijklmnopqrstuvwxyz", 12).len == 12,
        "truncateMiddle length")
  check(truncateMiddle("short", 12) == "short", "truncateMiddle leaves short text")
  echo testsRun - testsFailed, " of ", testsRun, " checks passed"
  testsFailed

proc selfTestGui(): int =
  initLuts()
  initFont()
  let before = testsFailed
  # --- the font covers every printable ASCII character, and no two characters look the same
  for c in ' ' .. '~':
    check(glyphKnown[ord(c)], "font has " & $c)
  block:
    var twins = ""
    for a in '!' .. '~':
      for b in char(ord(a) + 1) .. '~':
        if glyphBits[ord(a)] == glyphBits[ord(b)]: twins.add $a & $b & " "
    check(twins.len == 0, "font glyphs that look the same: " & twins)
  check(textWidth("ab", 2) == 22 and textWidth("", 2) == 0 and textWidth("\xC3\xA9", 3) == 15, "textWidth")
  block:
    var fb: Fb
    fb.resizeFb(20, 20)
    fb.drawText(3, 4, "A", 2)                    # row 0 of the glyph is 01110: columns 1..3 lit
    check(fb.px[4 * 20 + 5] == White and fb.px[4 * 20 + 10] == White and fb.px[4 * 20 + 4] == 0 and
          fb.px[4 * 20 + 11] == 0 and fb.px[5 * 20 + 7] == White and fb.px[6 * 20 + 3] == White,
          "drawText puts the glyph where it belongs")
    fb.px[0] = 0xFF808080'u32
    fb.px[1] = 0xFFFFFFFF'u32
    fb.dimRect(0, 0, 2, 1)
    check(fb.px[0] == 0xFF404040'u32 and fb.px[1] == 0xFF7F7F7F'u32, "dimRect halves every channel")
  # --- key names for --keys
  block:
    let k = keyFromName("shift+right")
    check(k.ok and k.sym == KeyRight and k.mods == ModShift, "key name shift+right")
    let c = keyFromName("Ctrl+Left")
    check(c.ok and c.sym == KeyLeft and c.mods == ModCtrl, "key name Ctrl+Left")
    check(keyFromName("r").sym == ord('r') and keyFromName("enter").sym == KeyEnter and not keyFromName("banana").ok,
          "key names")
  # --- the application's logic on an image that is only a size
  proc fakeApp(w, h: int): App =
    result = newApp()
    result.img = Image(srcPath: "C:/pics/photo.png", w: w, h: h, bands: 3, levelCount: 1)
    discard result.fitToImage()
    result.centerRect()
  block:
    let a = fakeApp(6000, 4000)
    check(a.outRes == (1920, 1080) and a.n == 1, "default output is 1920 x 1080 at 1 x 1")
    check(a.mainLines()[0] == "[R] 1920 x 1080 -> 1920 x 1080  Full HD", "main line 1 at n = 1: " & a.mainLines()[0])
    check(a.mainLines()[1] == "[S] 1 x 1 -> 1 x 1  pixel-perfect", "main line 2: " & a.mainLines()[1])
    check(a.mainLines()[2].startsWith("[M] by mean"), "main line 3")
    check(a.rx == (6000 - 1920) div 2 and a.ry == (4000 - 1080) div 2, "the rectangle starts centred")
    # R opens the resolution list at the current entry; Down x2 and Enter chooses 2560 x 1440 -> keeps the centre
    a.handleKey(ord('r'), 0, false)
    check(a.menu == mkResolution and a.menuSel == 3, "R opens the resolution list on the current size")
    for _ in 0 ..< 4: a.handleKey(KeyDown, 0, false)     # 1920x1200, 2048x1080, 2560x1080, 2560x1440
    a.handleKey(KeyEnter, 0, false)
    check(a.menu == mkNone and a.outRes == (2560, 1440), "choosing from the list changes the output size")
    check(abs((a.rx + 2560 div 2) - 3000) <= 1 and abs((a.ry + 1440 div 2) - 2000) <= 1, "the rectangle keeps its centre")
    # S: choose 2 x 2 by digit; 2560 x 1440 x 2 = 5120 > 6000? no: 5120 fits, 2880 fits
    a.handleKey(ord('s'), 0, false)
    a.handleKey(ord('2'), 0, false)
    check(a.n == 2 and a.mainLines()[0] == "[R] 5120 x 2880 -> 2560 x 1440  QHD", "scale 2: " & a.mainLines()[0])
    check(a.mainLines()[1] == "[S] 2 x 2 -> 1 x 1  pixel-group-perfect", "main line 2 at n = 2")
    # a scale that does not fit is refused and says so
    a.handleKey(ord('s'), 0, false)
    a.handleKey(ord('3'), 0, false)
    check(a.n == 2 and a.status.contains("needs more pixels"), "3 x 3 does not fit and is refused")
    a.handleKey(KeyEsc, 0, false)
    # M: sampling
    a.handleKey(ord('m'), 0, false)
    a.handleKey(KeyDown, 0, false)
    a.handleKey(KeyEnter, 0, false)
    check(a.mode == smMedian and a.mainLines()[2] == "[M] by channelwise median", "sampling switches to the channelwise median")
    a.handleKey(ord('m'), 0, false)
    a.handleKey(KeyEnd, 0, false)
    a.handleKey(KeyEnter, 0, false)
    check(a.mode == smAll and a.mainLines()[2] == "[M] by all  (10 images)", "the last sampling choice is all: " & a.mainLines()[2])
    a.mode = smMedian
    # arrows: one pixel, one output pixel with Ctrl, a quarter of the rectangle with Shift; clamped at the edges
    let (x0, y0) = (a.rx, a.ry)
    a.handleKey(KeyRight, 0, false)
    a.handleKeyUp(KeyRight)
    check(a.rx == x0 + 1 and a.ry == y0, "Right moves one source pixel")
    a.handleKey(KeyDown, ModCtrl, false)
    check(a.ry == y0 + 2, "Ctrl+Down moves one output pixel (n source pixels)")
    a.handleKey(KeyLeft, ModShift, false)
    check(a.rx == 0, "Shift+Left is clamped at the edge (a quarter would be 1280 pixels, 441 are left)")
    for _ in 0 ..< 10: a.handleKey(KeyLeft, ModShift, false)
    check(a.rx == 0, "the rectangle stops at the left edge")
    for _ in 0 ..< 30: a.handleKey(KeyDown, ModShift, false)
    check(a.ry == 4000 - 2880, "the rectangle stops at the bottom edge")
    a.handleKey(ord('c'), 0, false)
    check(a.rx == (6000 - 5120) div 2, "C centres the rectangle")
    # Tab hides the panels, Z toggles the exact camera
    a.handleKey(KeyTab, 0, false)
    a.handleKey(ord('z'), 0, false)
    check(not a.uiVisible and a.exact, "Tab hides the UI, Z zooms to the rectangle")
  block:   # the sampling list: the modes in the order asked for, "all" last
    let a = fakeApp(6000, 4000)
    a.menu = mkSampling
    let items = a.menuItems()
    var labels: seq[string]
    for it in items: labels.add it.split("  ")[0]
    check(labels == @["mean", "channelwise median", "channelwise max", "channelwise min", "colorwise random sample",
                      "channelwise random samples", "top left", "colorwise farthest from mean",
                      "channelwise farthest from mean", "channelwise farthest from median", "all"],
          "the sampling list: " & $labels)
    a.menu = mkNone
  block:   # a held key repeats only when it moves something
    let a = fakeApp(6000, 4000)
    a.handleKey(ord('s'), 0, true)
    check(a.menu == mkNone, "a repeated S does not reopen the list")
    a.handleKey(ord('s'), 0, false)
    a.handleKey(KeyDown, 0, true)
    check(a.menu == mkScale and a.menuSel == 1, "a repeated Down keeps moving in the list")
    a.handleKey(KeyEnter, 0, true)
    check(a.menu == mkScale, "a repeated Enter does not choose")
    a.handleKey(KeyEsc, 0, false)
  block:   # an image that is still being converted is not swapped while an export reads the old one
    let a = fakeApp(6000, 4000)
    let first = a.img
    let ex = Exporter(job: cast[ptr ExportJob](allocShared0(sizeof(ExportJob))))   # an export that is still running
    a.exporter = ex
    a.prep = Prep(state: psDone, image: Image(srcPath: "b.png", w: 4000, h: 3000, bands: 3, levelCount: 1))
    a.pollWork()
    check(a.img == first and a.prep != nil and a.exporter == ex, "the new image waits for the export to finish")
    a.exporter = nil
    deallocShared(ex.job)
  block:   # Shift moves a quarter of the rectangle
    let a = fakeApp(20000, 12000)
    let (x0, y0) = (a.rx, a.ry)
    a.handleKey(KeyRight, ModShift, false)
    a.handleKey(KeyUp, ModShift, false)
    check(a.rx == x0 + 1920 div 4 and a.ry == y0 - 1080 div 4, "Shift+arrows move a quarter of the rectangle")
  block:   # custom size: W and H appear and are typed
    let a = fakeApp(6000, 4000)
    a.handleKey(ord('w'), 0, false)
    check(a.edit == efNone and a.status.contains("Custom"), "W does nothing until Custom is chosen")
    a.handleKey(ord('r'), 0, false)
    a.handleKey(KeyEnd, 0, false)
    a.handleKey(KeyEnter, 0, false)
    check(a.resIdx == CustomIndex and a.mainLines().len == 4 and a.mainLines()[3] == "[W] width 1920    [H] height 1080",
          "Custom shows the width and height controls: " & $a.mainLines())
    a.handleKey(ord('w'), 0, false)
    a.handleText("77")                                 # typing over the untouched value replaces it
    check(a.editBuf == "77", "the first digit typed replaces the number shown")
    for _ in 0 ..< 4: a.handleKey(KeyBackspace, 0, false)
    a.handleText("2")
    a.handleText("x")                                  # only digits are accepted
    a.handleText("400")
    check(a.editBuf == "2400" and a.mainLines()[3].contains("2400_"), "typing digits into the width")
    a.handleKey(KeyTab, 0, false)
    check(a.customW == 2400 and a.edit == efHeight, "Tab applies the width and moves to the height")
    for _ in 0 ..< 4: a.handleKey(KeyBackspace, 0, false)
    a.handleText("900")
    a.handleKey(KeyEnter, 0, false)
    check(a.edit == efNone and a.outRes == (2400, 900) and a.mainLines()[0] == "[R] 2400 x 900 -> 2400 x 900  custom",
          "Enter applies the height: " & a.mainLines()[0])
    a.handleKey(ord('h'), 0, false)
    a.handleText("9000")                               # 9000 does not fit a 4000 pixel high image
    a.handleKey(KeyEnter, 0, false)
    check(a.edit == efHeight and a.customH == 900 and a.status.contains("needs more pixels"), "a custom size that does not fit is refused")
    a.handleKey(KeyEsc, 0, false)
    check(a.edit == efNone, "Esc leaves the editing")
  block:   # an image too small for the current choice reduces it
    let a = newApp()
    a.n = 4
    a.img = Image(srcPath: "x.png", w: 2600, h: 1500, bands: 3, levelCount: 1)
    check(a.fitToImage() and a.n == 1 and a.outRes == (1920, 1080), "n shrinks until the rectangle fits")
    let b = newApp()
    b.img = Image(srcPath: "x.png", w: 1000, h: 600, bands: 3, levelCount: 1)
    check(b.fitToImage() and b.resIdx == CustomIndex and b.outRes == (1000, 600), "a tiny image gets a custom size")
    let c = newApp()
    c.img = Image(srcPath: "x.png", w: 1500, h: 900, bands: 3, levelCount: 1)
    check(c.fitToImage() and c.outRes == (1366, 768), "the largest named size that fits is chosen")
  # --- the rendered picture and outline, on a real (tiny) memory-mapped image
  let scratch = getTempDir() / ("wallcrop-selftest-" & $getCurrentProcessId())
  createDir(scratch)
  cacheRootOverride = scratch / "cache"
  try:
    let w = 100
    let h = 80
    var raw = newSeq[uint8](w * h * 3)
    for y in 0 ..< h:
      for x in 0 ..< w:
        raw[(y * w + x) * 3] = uint8((x * 5) mod 256)
        raw[(y * w + x) * 3 + 1] = uint8((y * 9) mod 256)
        raw[(y * w + x) * 3 + 2] = uint8(if (x + y) mod 7 == 0: 255 else: 20)
    let dir = scratch / "img"
    createDir(dir)
    writeFile(dir / "l0.raw", cast[string](raw))
    let img = Image(srcPath: scratch / "tiny.png", dir: dir, w: w, h: h, bands: 3, levelCount: levelCountFor(w, h))
    check(mapLevel(img, 0), "map a raw file")
    let a = newApp()
    a.img = img
    a.changeOutput(CustomIndex, 64, 48, 1, smMean)
    a.rx = 10
    a.ry = 8
    a.uiVisible = false
    a.exact = true
    var fb: Fb
    fb.resizeFb(256, 192)                             # 4 screen pixels per source pixel: the camera is exact
    a.renderFrame(fb)
    var wrong = 0
    var outlineWrong = 0
    for y in 0 ..< 192:
      for x in 0 ..< 256:
        let sx = 10 + x div 4
        let sy = 8 + y div 4
        let o = (sy * w + sx) * 3
        let want = 0xFF000000'u32 or (uint32(raw[o]) shl 16) or (uint32(raw[o + 1]) shl 8) or uint32(raw[o + 2])
        let onOutline = x == 0 or y == 0 or x == 255 or y == 191
        if onOutline:
          let expect = if outlineIsWhite(int(raw[o]), int(raw[o + 1]), int(raw[o + 2])): White else: Black
          if fb.px[y * 256 + x] != expect: inc outlineWrong
        elif fb.px[y * 256 + x] != want: inc wrong
    check(wrong == 0, "exact camera draws each source pixel as a 4 x 4 block (" & $wrong & " wrong)")
    check(outlineWrong == 0, "the outline is white or black by contrast with the picture (" & $outlineWrong & " wrong)")
    # exact view in a window of another shape: the margins are covered with 75% opaque black, the rectangle is not
    var tall: Fb
    tall.resizeFb(256, 256)                           # the rectangle is 256 x 192 here: 32 rows of margin above and below
    a.renderFrame(tall)
    var marginWrong = 0
    var insideWrong = 0
    for y in 0 ..< 256:
      for x in 0 ..< 256:
        let sx = 10 + x div 4
        let sy = clampInt(8 + int(floor((float(y) + 0.5 - 32.0) / 4.0)), 0, h - 1)
        let o = (sy * w + sx) * 3
        let full = 0xFF000000'u32 or (uint32(raw[o]) shl 16) or (uint32(raw[o + 1]) shl 8) or uint32(raw[o + 2])
        let quarter = 0xFF000000'u32 or ((uint32(raw[o]) shr 2) shl 16) or ((uint32(raw[o + 1]) shr 2) shl 8) or (uint32(raw[o + 2]) shr 2)
        if y < 32 or y >= 224:
          if tall.px[y * 256 + x] != quarter: inc marginWrong
        elif x > 0 and x < 255 and y > 32 and y < 223 and tall.px[y * 256 + x] != full: inc insideWrong
    check(marginWrong == 0, "margins of the exact view are a quarter as bright (75% black) (" & $marginWrong & " wrong)")
    check(insideWrong == 0, "the rectangle itself is not dimmed (" & $insideWrong & " wrong)")
    a.exact = false
    a.renderFrame(fb)
    let cam = makeCam(256, 192, 64, 48, false)
    check(cam.left == 26 and cam.top == 19 and cam.rwS == 205, "normal camera is zoomed out from the rectangle")
    check(fb.px[(cam.top + cam.rhS div 2) * 256 + cam.left] in [White, Black] and
          fb.px[(cam.top + 3) * 256 + cam.left + 3] != White, "outline sits on the rectangle edge, picture inside")
    a.img = nil                                       # an empty window still draws the hint
    a.uiVisible = true
    a.renderFrame(fb)
    var lit = 0
    for p in fb.px:
      if p == White: inc lit
    check(lit > 100, "the window without an image shows the open-an-image hint")
    # --- the pyramid, built on its thread
    let big = 700
    var raw2 = newSeq[uint8](big * big * 3)
    for i in 0 ..< raw2.len: raw2[i] = uint8(rand(255))
    let dir2 = scratch / "pyr"
    createDir(dir2)
    writeFile(dir2 / "l0.raw", cast[string](raw2))
    let img2 = Image(srcPath: scratch / "p.png", dir: dir2, w: big, h: big, bands: 3, levelCount: levelCountFor(big, big))
    check(img2.levelCount == 2 and mapLevel(img2, 0), "a 700 x 700 image has two levels")
    let pyr = startPyramid(img2)
    check(pyr != nil, "pyramid starts")
    while not pyramidDone(pyr): sleep(10)
    check(pyr.job.ready.load() == 2, "pyramid finished")
    stopPyramid(pyr)
    check(mapLevel(img2, 1) and img2.levels[1].w == 350, "level 1 is mapped")
    var bad = 0
    for (x, y) in [(0, 0), (349, 349), (100, 200), (17, 3)]:
      for ch in 0 ..< 3:
        var s = 0
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)]: s += int(raw2[((2 * y + dy) * big + 2 * x + dx) * 3 + ch])
        if int(pixelsOf(img2, 1)[(y * 350 + x) * 3 + ch]) != (s + 2) div 4: inc bad
    check(bad == 0, "pyramid level 1 holds 2 x 2 means")
    check(startPyramid(img2) == nil, "a finished pyramid is not built again")
    closeImage(img2)
    closeImage(img)
    # --- through vips and back: raw -> PNG -> cache -> same bytes; then export through the real path
    if vipsTool("vips").len == 0:
      echo "  (vips not found: skipping the vips round trip)"
    else:
      let pw = 37
      let ph = 29
      var rgba = newSeq[uint8](pw * ph * 4)
      for i in 0 ..< rgba.len: rgba[i] = uint8(rand(255))
      writeFile(scratch / "rgba.raw", cast[string](rgba))
      check(writePngWithVips(scratch / "rgba.raw", pw, ph, 4, scratch / "rgba.png") == "", "vips writes a PNG")
      let (opened, err, cached) = openBlocking(scratch / "rgba.png")
      check(err == "" and opened != nil and not cached and opened.bands == 4 and opened.w == pw and opened.h == ph,
            "vips converts the PNG into a cache: " & err)
      if opened != nil:
        var same = true
        for i in 0 ..< rgba.len:
          if pixelsOf(opened, 0)[i] != rgba[i]: same = false
        check(same, "the cache holds exactly the PNG's pixels (RGBA)")
        let reopened = openCached(scratch / "rgba.png", cacheDirFor(scratch / "rgba.png"))
        check(reopened != nil and reopened.w == pw, "the second open is a cache hit")
        closeImage(reopened)
        check(exportToPng(opened, 1, 2, 5, 4, 3, smMedian, scratch / "out.png") == "", "export through vips")
        let (back, err2, _) = openBlocking(scratch / "out.png")
        check(err2 == "" and back != nil and back.w == 5 and back.h == 4 and back.bands == 4, "the exported PNG is 5 x 4 with alpha")
        if back != nil:
          var ok = true
          for oy in 0 ..< 4:
            for ox in 0 ..< 5:
              for ch in 0 ..< 4:
                var vals: seq[int]
                for dy in 0 ..< 3:
                  for dx in 0 ..< 3: vals.add int(rgba[((2 + oy * 3 + dy) * pw + 1 + ox * 3 + dx) * 4 + ch])
                vals.sort()
                if int(pixelsOf(back, 0)[(oy * 5 + ox) * 4 + ch]) != vals[4]: ok = false
          check(ok, "the exported PNG holds the 3 x 3 medians")
          closeImage(back)
        block:     # "all": one image per mode, made by the export thread, each equal to its reference
          let a = newApp()
          a.img = opened
          a.saveDir = scratch / "allout"
          createDir(a.saveDir)
          a.changeOutput(CustomIndex, 5, 4, 2, smAll)
          a.rx = 3
          a.ry = 2
          a.startExport()
          check(a.exporter != nil and a.exporter.count == 10, "'all' starts ten exports")
          while a.exporter != nil:
            a.pollWork()
            sleep(20)
          var made = 0
          var wrong = 0
          for m in SingleModes:
            let file = a.saveDir / exportName("rgba", 3, 2, 5, 4, 2, m)
            if fileExists(file):
              inc made
              let (img2, err3, _) = openBlocking(file)
              if err3.len > 0 or img2 == nil: inc wrong
              else:
                for oy in 0 ..< 4:
                  for ox in 0 ..< 5:
                    let want = refBlock(rgba, pw, 4, 3 + ox * 2, 2 + oy * 2, 2, m)
                    for ch in 0 ..< 4:
                      if int(pixelsOf(img2, 0)[(oy * 5 + ox) * 4 + ch]) != want[ch]: inc wrong
                closeImage(img2)
          check(made == 10 and wrong == 0, "'all' saved ten images that match their modes (" & $made & " files, " & $wrong & " wrong values)")
          a.changeOutput(CustomIndex, 5, 4, 1, smAll)
          a.startExport()
          check(a.exporter != nil and a.exporter.count == 1, "'all' at 1 x 1 makes one image: every mode would be the same")
          while a.exporter != nil:
            a.pollWork()
            sleep(20)
        check(exportToPng(opened, 30, 0, 5, 4, 3, smMean, scratch / "bad.png") == "the rectangle is not inside the image",
              "a rectangle outside the image is refused")
        closeImage(opened)
  finally:
    cacheRootOverride = ""
    try: removeDir(scratch)
    except OSError: discard
  echo testsRun - testsFailed, " of ", testsRun, " checks passed (including the interface)"
  testsFailed - before

proc snapshotCli(args: seq[string]): int =
  ## wallcrop --snapshot out.png [--size 1280x720] [--image file] [--keys "r down enter"]
  var outPath, imagePath, keys = ""
  var w = 1280
  var h = 720
  var i = 0
  try:
    while i < args.len:
      case args[i]
      of "--size", "--image", "--keys":
        if i + 1 >= args.len: raise newException(ValueError, "missing value after " & args[i])
        let value = args[i + 1]
        if args[i] == "--size":
          let parts = value.split('x')
          w = parseInt(parts[0])
          h = parseInt(parts[1])
        elif args[i] == "--image": imagePath = value
        else: keys = value
        inc i
      else: outPath = args[i]
      inc i
  except ValueError, IndexDefect:
    echo "bad arguments: ", getCurrentExceptionMsg()
    outPath = ""
  if outPath.len == 0:
    echo "usage: --snapshot out.png [--size WxH] [--image file] [--keys \"r down enter\"]"
    return 2
  initLuts()
  initFont()
  let app = newApp()
  if imagePath.len > 0:
    let (img, err, _) = openBlocking(imagePath)
    if err.len > 0:
      echo "error: ", err
      return 1
    app.finishOpen(img)
    while app.pyr != nil:
      refreshLevels(app)
      sleep(20)
    app.status = ""
  app.applyKeyScript(keys)
  var fb: Fb
  fb.resizeFb(w, h)
  app.renderFrame(fb)
  var raw = newSeq[uint8](w * h * 3)
  for k in 0 ..< w * h:
    raw[k * 3] = uint8((fb.px[k] shr 16) and 255)
    raw[k * 3 + 1] = uint8((fb.px[k] shr 8) and 255)
    raw[k * 3 + 2] = uint8(fb.px[k] and 255)
  createDir(appDir())
  let tmp = appDir() / ("snapshot-" & $getCurrentProcessId() & ".raw")
  writeFile(tmp, cast[string](raw))
  let err = writePngWithVips(tmp, w, h, 3, outPath)
  try: removeFile(tmp)
  except OSError: discard
  if err.len > 0:
    echo "error: ", err
    return 1
  echo "wrote ", outPath
  app.closeCurrentImage()
  0

proc exportCli(args: seq[string]): int =
  ## wallcrop --export <image> <x> <y> <outW> <outH> <n> <mode> <out.png>   (mode: see --help; "all" writes one
  ## file per mode, with the mode's name before the extension)
  if args.len != 8:
    echo "usage: --export <image> <x> <y> <outW> <outH> <n> <mode> <out.png>"
    return 2
  let (modeOk, mode) = parseMode(args[6])
  if not modeOk:
    echo "unknown sampling mode ", args[6], "; the modes are: ", ModeFileNames.join(" ")
    return 2
  let t0 = getMonoTime()
  let (img, err, cached) = openBlocking(args[0])
  if err.len > 0:
    echo "error: ", err
    return 1
  echo "image ", img.w, " x ", img.h, " x ", img.bands, (if cached: " (cache hit)" else: " (converted)"), " in ",
    (getMonoTime() - t0).inMilliseconds, " ms"
  let t1 = getMonoTime()
  let n = parseInt(args[5])
  let modes = if mode == smAll and n > 1: @SingleModes else: @[(if mode == smAll: smMean else: mode)]
  for m in modes:
    var target = args[7]
    if modes.len > 1:
      let (dir, name, ext) = splitFile(args[7])
      target = dir / (name & "_" & ModeFileNames[m] & ext)
    let e = exportToPng(img, parseInt(args[1]), parseInt(args[2]), parseInt(args[3]), parseInt(args[4]),
                        n, m, target)
    if e.len > 0:
      echo "error: ", e
      return 1
    echo "exported (", ModeLabels[m], ") in ", (getMonoTime() - t1).inMilliseconds, " ms -> ", target
  0

proc benchCli(args: seq[string]): int =
  ## wallcrop --bench image : time the drawing of 1920 x 1080 frames while the rectangle moves
  if args.len != 1:
    echo "usage: --bench <image>"
    return 2
  initLuts()
  initFont()
  let (img, err, _) = openBlocking(args[0])
  if err.len > 0:
    echo "error: ", err
    return 1
  let app = newApp()
  app.finishOpen(img)
  while app.pyr != nil:
    refreshLevels(app)
    sleep(20)
  var fb: Fb
  fb.resizeFb(1920, 1080)
  for (label, resIdx, n, exact) in [("n=1, exact view (1:1 pixels)", 3, 1, true), ("n=1, normal camera", 3, 1, false),
                                    ("n=2, normal camera", 3, 2, false), ("n=4, normal camera", 3, 4, false),
                                    ("n=8, normal camera", 3, 8, false), ("n=4, exact view", 3, 4, true)]:
    if not app.fits(Resolutions[resIdx].w, Resolutions[resIdx].h, n): continue
    app.changeOutput(resIdx, 1920, 1080, n, smMean)
    app.exact = exact
    app.centerRect()
    var worst = 0.0
    let t0 = getMonoTime()
    let frames = 120
    for i in 0 ..< frames:
      app.rx = clampInt(app.rx + 7, 0, img.w - Resolutions[resIdx].w * n)
      app.ry = clampInt(app.ry + 5, 0, img.h - Resolutions[resIdx].h * n)
      let f0 = getMonoTime()
      app.renderFrame(fb)
      worst = max(worst, float((getMonoTime() - f0).inMicroseconds) / 1000.0)
    let total = float((getMonoTime() - t0).inMicroseconds) / 1000.0
    echo label, ": ", formatFloat(total / frames.float, ffDecimal, 2), " ms per frame (slowest ", formatFloat(worst, ffDecimal, 2), " ms)"
  app.closeCurrentImage()
  0

const HelpText = """wallcrop - crop a huge image to wallpapers, pixel-perfect or pixel-group-perfect

  nim r -d:release dev/tools/wallcrop.nim [image]     open the window (the last image if none is given)
  ... --selftest                                      run the checks
  ... --export image x y outW outH n MODE out.png     export without the window (modes below; all = one file per mode)
  ... --snapshot out.png [--size WxH] [--image file] [--keys "r down enter"]   draw one frame to a PNG
  ... --help

Keys:  R output size   S scale (n x n -> 1 x 1)   M sampling mode   W / H custom width / height
       O open an image (or drop one on the window)   F save folder   E export   Z exact view
       arrows move the rectangle (hold to speed up; Ctrl = 1 output pixel, Shift = a quarter of it)
       C centre   Tab hide the text   F11 fullscreen   Ctrl+Q quit

Sampling modes (what one output pixel is made of, from its n x n block; --export names in brackets):
  mean (mean)   channelwise median (median)   channelwise max (max)   channelwise min (min)
  colorwise random sample (random-color)   channelwise random samples (random-channels)   top left (top-left)
  colorwise farthest from mean (far-mean-color)   channelwise farthest from mean (far-mean-channels)
  channelwise farthest from median (far-median-channels)   all (all: one image per mode above)
"""

proc main() =
  let args = commandLineParams()
  if args.len > 0 and args[0] == "--selftest":
    let failed = selfTest() + selfTestGui()
    quit(if failed == 0: 0 else: 1)
  elif args.len > 0 and args[0] == "--export": quit(exportCli(args[1 .. ^1]))
  elif args.len > 0 and args[0] == "--snapshot": quit(snapshotCli(args[1 .. ^1]))
  elif args.len > 0 and args[0] == "--bench": quit(benchCli(args[1 .. ^1]))
  elif args.len > 0 and args[0] in ["--help", "-h"]:
    echo HelpText
    echo "cache folder: ", cacheRoot()
    echo "libvips:      ", (let v = vipsTool("vips"); if v.len > 0: v else: "NOT FOUND (set VipsDir)")
  else:
    var path = ""
    for a in args:
      if not a.startsWith("--"): path = a
    quit(runWindow(path))

when isMainModule:
  main()
