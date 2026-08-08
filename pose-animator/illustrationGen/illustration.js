/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { Bone, allPartNames, Skeleton } from './skeleton.js';
import { MathUtils } from '../utils/mathUtils.js';
import { SVGUtils } from '../utils/svgUtils.js';
import { ColorUtils } from '../utils/colorUtils.js';

const allPartNamesMap = {};
allPartNames.forEach(name => allPartNamesMap[name] = 1);

const MIN_CONFIDENCE_PATH_SCORE = 0.3;
const MIN_HAND_SKINNING_DISTANCE = 0.75;
const MAX_HAND_BONE_INFLUENCES = 4;
const HAND_MIN_SCORE = 0.001;
const HAND_FULL_DETAIL_SCORE = 0.75;
const FLIP_HELD_DETAIL_FACTOR = 0.4;
const NATIVE_HAND_PATH_PREFIXES = {
    nativeLeftHand: 'left',
    nativeRightHand: 'right',
};
const PROCEDURAL_HAND_GROUP_PREFIX = 'illustration_generated_';

function getExplicitHandSide(path) {
    const name = String(path && path.name ? path.name : '');
    const prefix = Object.keys(NATIVE_HAND_PATH_PREFIXES).find(candidate => name.startsWith(candidate));
    return prefix ? NATIVE_HAND_PATH_PREFIXES[prefix] : null;
}

function getProceduralHandSide(path) {
    let item = path;
    while (item) {
        const name = String(item.name || '');
        if (name.startsWith(`${PROCEDURAL_HAND_GROUP_PREFIX}left_hand`)) {
            return 'left';
        }
        if (name.startsWith(`${PROCEDURAL_HAND_GROUP_PREFIX}right_hand`)) {
            return 'right';
        }
        item = item.parent;
    }
    return null;
}

function getDetailedHandAlpha(hand) {
    if (!hand || Number(hand.score || 0) < HAND_MIN_SCORE) return 0;
    const visibleLandmarks = Array.isArray(hand.keypoints)
        ? hand.keypoints.filter(kp => kp && kp.position && Number(kp.score || 0) > 0).length
        : 0;
    return visibleLandmarks >= 6 ? 1 : 0;
}

// Represents a skinned illustration.
export class PoseIllustration {
    constructor(scope) {
        this.scope = scope;
        this.frames = [];
        this.preferDetailedHands = true;
        this.useProceduralHands = false;
    }

    hasProceduralHandRig() {
        return ['left', 'right'].every(side => this.skinnedPaths.some(path => (
            path.isProceduralHandPath && path.handSide === side
        )));
    }

    updateSkeleton(pose, face, hands = null) {
        this.pose = pose;
        this.face = face;
        this.hands = hands;
        this.skeleton.update(pose, face, hands);
        if (!this.skeleton.isValid) {
            return;
        }

        let getConfidenceScore = (p) => {
            return Object.keys(p.skinning).reduce((totalScore, boneName) => {
                let bt = p.skinning[boneName];
                return totalScore + bt.bone.score * bt.weight;
            }, 0);
        }

        this.skinnedPaths.forEach(skinnedPath => {
            let confidenceScore = 0;
            skinnedPath.segments.forEach(seg => {
                // Compute confidence score.
                confidenceScore += getConfidenceScore(seg.point);
                // Compute new positions for curve point and handles.
                seg.point.currentPosition = Skeleton.getCurrentPosition(seg.point);
                if (seg.handleIn) {
                    seg.handleIn.currentPosition = Skeleton.getCurrentPosition(seg.handleIn);
                }
                if (seg.handleOut) {
                    seg.handleOut.currentPosition = Skeleton.getCurrentPosition(seg.handleOut);
                }
            });
            skinnedPath.confidenceScore = confidenceScore / (skinnedPath.segments.length || 1);
        });
    }

    draw() {
        if (!this.skeleton.isValid) {
            return;
        }
        let scope = this.scope;
        // Add paths
        this.skinnedPaths.forEach(skinnedPath => {
            const detailedHand = skinnedPath.handSide && this.hands && this.hands[skinnedPath.handSide];
            const detailedHandAlpha = this.preferDetailedHands ? getDetailedHandAlpha(detailedHand) : 0;
            // Do not render paths with low confidence scores.
            const minConfidence = skinnedPath.isHandPath ? 0.05 : MIN_CONFIDENCE_PATH_SCORE;
            if (!skinnedPath.confidenceScore || skinnedPath.confidenceScore < minConfidence) {
                return;
            }
            let path = new scope.Path({
                fillColor: skinnedPath.fillColor,
                strokeColor: skinnedPath.strokeColor,
                strokeWidth: skinnedPath.strokeWidth,
                closed: skinnedPath.closed,
            });
            if (skinnedPath.isHandPath) {
                if (skinnedPath.isProceduralHandPath) {
                    path.opacity = this.useProceduralHands ? detailedHandAlpha : 0;
                } else {
                    // Native hands stay opaque when detailed tracking is absent.
                    path.opacity = detailedHandAlpha > 0 ? 0 : 1;
                }
            }
            skinnedPath.segments.forEach(seg => {
                path.addSegment(seg.point.currentPosition, 
                    seg.handleIn ? seg.handleIn.currentPosition.subtract(seg.point.currentPosition) : null,
                    seg.handleOut ? seg.handleOut.currentPosition.subtract(seg.point.currentPosition) : null);
            });
            if (skinnedPath.closed) {
                path.closePath();
            }
            scope.project.activeLayer.addChild(path);
        });
    }

    drawRestPose() {
        let scope = this.scope;
        let group = new scope.Group();
        scope.project.activeLayer.addChild(group);

        this.skinnedPaths.forEach(skinnedPath => {
            let path = new scope.Path({
                fillColor: skinnedPath.fillColor,
                strokeColor: skinnedPath.strokeColor,
                strokeWidth: skinnedPath.strokeWidth,
                closed: skinnedPath.closed,
            });
            skinnedPath.segments.forEach(seg => {
                path.addSegment(
                    seg.point.position,
                    seg.handleIn ? seg.handleIn.position.subtract(seg.point.position) : null,
                    seg.handleOut ? seg.handleOut.position.subtract(seg.point.position) : null
                );
            });
            if (skinnedPath.closed) {
                path.closePath();
            }
            group.addChild(path);
        });

        if (group.bounds && group.bounds.width > 0 && group.bounds.height > 0) {
            const targetWidth = scope.view.size.width * 0.55;
            const targetHeight = scope.view.size.height * 0.78;
            const scale = Math.min(targetWidth / group.bounds.width, targetHeight / group.bounds.height);
            if (isFinite(scale) && scale > 0) {
                group.scale(scale);
            }
            group.position = new scope.Point(scope.view.size.width * 0.5, scope.view.size.height * 0.5);
        }
    }

    debugDraw() {
        let scope = this.scope;
        let group = new scope.Group();
        scope.project.activeLayer.addChild(group);
        let drawCircle = (p, opt = {}) => {
            group.addChild(new scope.Path.Circle({
                center: [p.x, p.y],
                radius: opt.radius || 2,
                fillColor: opt.fillColor || 'red',
            }));
        }
        let drawLine = (p0, p1, opt = {}) => {
            group.addChild(new scope.Path({
                segments: [p0, p1],
                strokeColor: opt.strokeColor || 'red',
                strokeWidth: opt.strokeWidth || 1
            }));
        }
        // Draw skeleton.
        this.skeleton.debugDraw(scope);
        // Draw curve and handles.
        this.skinnedPaths.forEach(skinnedPath => {
            skinnedPath.segments.forEach(seg => {
                // Color represents weight influence from bones.
                let color = new scope.Color(0);
                Object.keys(seg.point.skinning).forEach((boneName) => {
                    let bt = seg.point.skinning[boneName];
                    ColorUtils.addRGB(color, 
                        bt.weight * bt.bone.boneColor.red, 
                        bt.weight * bt.bone.boneColor.green, 
                        bt.weight * bt.bone.boneColor.blue);
                        let anchor = bt.bone.kp0.currentPosition.multiply(1 - bt.transform.anchorPerc).add(bt.bone.kp1.currentPosition.multiply(bt.transform.anchorPerc));
                        drawLine(anchor, seg.point.currentPosition, {strokeColor: 'blue', strokeWidth: bt.weight});
                });

                drawCircle(seg.point.currentPosition, {fillColor: color});
                drawCircle(seg.handleIn.currentPosition, {fillColor: color});
                drawLine(seg.point.currentPosition, seg.handleIn.currentPosition, {strokeColor: color});
                drawCircle(seg.handleOut.currentPosition, {fillColor: color}, {strokeColor: color});
                drawLine(seg.point.currentPosition, seg.handleOut.currentPosition);
            });
        });
    }

    debugDrawLabel(scope) {
        this.skeleton.debugDrawLabels(scope);
    }

    bindSkeleton(skeleton, skeletonScope) {
        let items = skeletonScope.project.getItems({ recursive: true });
        items = items.filter(item => item.parent && item.parent.name && item.parent.name.startsWith('illustration'));
        this.skeleton = skeleton;
        this.skinnedPaths = [];

        // Only support rendering path and shapes for now.
        for (let i = 0; i < items.length; i++) {
            let item = items[i];
            if (SVGUtils.isGroup(item)) {
                this.bindGroup(item, skeleton);
            } else if (SVGUtils.isPath(item)) {
                this.bindPathToBones(item);
            } else if (SVGUtils.isShape(item)) {
                this.bindPathToBones(item.toPath());
            }
        }
    }

    bindGroup(group, skeleton) {
        let paths = [];
        let keypoints = {};
        let items = group.getItems({recursive: true});
        // Find all paths and included keypoints.
        items.forEach(item => {
            let partName = item.name ? allPartNames.find(partName => item.name.startsWith(partName)) : null;
            if (partName) {
                keypoints[partName] = {
                    position: item.bounds.center,
                    name: partName,
                };
            } else if (SVGUtils.isPath(item)) {
                paths.push(item);
            } else if (SVGUtils.isShape(item)) {
                paths.push(item.toPath());
            }
        });
        let secondaryBones = [];
        // Find all parent bones of the included keypoints.
        let parentBones = skeleton.bones.filter(bone => keypoints[bone.kp0.name] && keypoints[bone.kp1.name]);
        let nosePos = skeleton.bNose3Nose4.kp1.position;
        if (!parentBones.length) {
            return;
        }

        // Crete secondary bones for the included keypoints.
        parentBones.forEach(parentBone => {
            let kp0 = keypoints[parentBone.kp0.name];
            let kp1 = keypoints[parentBone.kp1.name];
            let secondaryBone = new Bone().set(kp0, kp1, parentBone.skeleton, parentBone.type);
            kp0.transformFunc = MathUtils.getTransformFunc(parentBone.kp0.position, nosePos, kp0.position);
            kp1.transformFunc = MathUtils.getTransformFunc(parentBone.kp1.position, nosePos, kp1.position);
            secondaryBone.parent = parentBone;
            secondaryBones.push(secondaryBone);
        });        
        skeleton.secondaryBones = skeleton.secondaryBones.concat(secondaryBones);
        paths.forEach(path => {
            this.bindPathToBones(path, secondaryBones);
        });
    }

    // Assign weights from bones for point.
    // Weight calculation is roughly based on linear blend skinning model.
    getWeights(point, bones) {
        let totalW = 0;
        let weights = {};
        const handOnly = bones.length > 0 && bones.every(bone => bone.type === 'hand');
        bones.forEach(bone => {
            let d = MathUtils.getClosestPointOnSegment(bone.kp0.position, bone.kp1.position, point)
                .getDistance(point);
            // Procedural joints can lie exactly on several bones. Clamp the
            // distance to prevent Infinity / Infinity during normalization.
            d = Math.max(d, handOnly ? MIN_HAND_SKINNING_DISTANCE : 1e-4);
            // Absolute weight = 1 / (distance * distance)
            let w = 1 / (d * d);
            weights[bone.name] = {
                value: w,
                bone: bone,
            }
        });

        let values = Object.values(weights).sort((v0, v1) => {
            return v1.value - v0.value;
        });
        if (handOnly) {
            // Keep each finger segment attached to its local chain instead of
            // blending it with every bone in the palm and other fingers.
            values = values.slice(0, MAX_HAND_BONE_INFLUENCES);
        }
        weights = {};
        totalW = 0;
        values.forEach(v => {
            weights[v.bone.name] = v;
            totalW += v.value;
        });
        if (totalW === 0) {
            // Point is outside of the influence zone of all bones. It will not be influence by any bone.
            return {};
        }

        // Normalize weights to sum up to 1.
        Object.values(weights).forEach(weight => {
            weight.value /= totalW;
        });

        return weights;
    }

    // Binds a path to bones by compute weight contribution from each bones for each path segment.
    // If selectedBones are set, bind directly to the selected bones. Otherwise auto select the bone group closest to each segment.
    bindPathToBones(path, selectedBones) {
        // Compute bone weights for each segment.
        let segs = path.segments.map(s => {
            // Check if control points are collinear.
            // If so, use the middle point's weight for all three points (curve point, handleIn, handleOut).
            // This makes sure smooth curves remain smooth after deformation.
            let collinear = MathUtils.isCollinear(s.handleIn, s.handleOut);
            let bones = selectedBones || this.skeleton.findBoneGroup(s.point);
            let weightsP = this.getWeights(s.point, bones);
            let segment = {
                point: this.getSkinning(s.point, weightsP),
            };
            // For handles, compute transformation in world space.
            if (s.handleIn) {
                let pHandleIn = s.handleIn.add(s.point);
                segment.handleIn = this.getSkinning(pHandleIn, collinear ? weightsP : this.getWeights(pHandleIn, bones));
            }
            if (s.handleOut) {
                let pHandleOut = s.handleOut.add(s.point);
                segment.handleOut = this.getSkinning(pHandleOut, collinear ? weightsP : this.getWeights(pHandleOut, bones));
            }
            return segment;
        });
        let handDominantSegments = 0;
        const handSideWeights = { left: 0, right: 0 };
        segs.forEach(segment => {
            const skinningWeights = Object.values(segment.point.skinning);
            const handWeight = skinningWeights.reduce((total, weight) => {
                return total + (weight.bone.type === 'hand' ? Number(weight.weight || 0) : 0);
            }, 0);
            if (handWeight < 0.6) return;
            handDominantSegments += 1;
            skinningWeights.forEach(weight => {
                if (weight.bone.type !== 'hand') return;
                const partName = weight.bone && weight.bone.kp0 ? weight.bone.kp0.name : '';
                if (partName.startsWith('leftHand')) {
                    handSideWeights.left += Number(weight.weight || 0);
                } else if (partName.startsWith('rightHand')) {
                    handSideWeights.right += Number(weight.weight || 0);
                }
            });
        });
        const proceduralHandSide = getProceduralHandSide(path);
        const explicitHandSide = getExplicitHandSide(path);
        const isProceduralHandPath = proceduralHandSide !== null;
        const isHandPath = explicitHandSide !== null || isProceduralHandPath
            || (segs.length > 0 && handDominantSegments / segs.length >= 0.65);
        let handSide = proceduralHandSide || explicitHandSide;
        if (isHandPath && !handSide) {
            handSide = handSideWeights.left >= handSideWeights.right ? 'left' : 'right';
        }
        this.skinnedPaths.push({
            segments: segs,
            fillColor: path.fillColor,
            strokeColor: path.strokeColor,
            strokeWidth: path.strokeWidth,
            closed: path.closed,
            isHandPath: isHandPath,
            isProceduralHandPath: isProceduralHandPath,
            handSide: handSide,
        });
    }

    getSkinning(point, weights) {
        let skinning = {};
        Object.keys(weights).forEach(boneName => {
            skinning[boneName] = {
                bone: weights[boneName].bone,
                weight: weights[boneName].value,
                transform: weights[boneName].bone.getPointTransform(point),
            };
        });
        return {
            skinning: skinning,
            position: point,
            currentPosition: new this.scope.Point(0, 0),
        }
    };
}
