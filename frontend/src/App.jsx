import { useState } from "react";

import {
    Container,
    Grid,
    Card,
    Box,
    Typography,
    Stack,
    Button,
    Chip,
    LinearProgress,
} from "@mui/material";

import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import AnalyticsIcon from "@mui/icons-material/Analytics";

import api from "./api/api";

export default function App() {

    const [file, setFile] =
        useState(null);

    const [loading,
        setLoading] =
        useState(false);

    const [originalImage,
        setOriginalImage] =
        useState(null);

    const [overlayImage,
        setOverlayImage] =
        useState(null);

    const [predictionMask,
        setPredictionMask] =
        useState(null);

    const [stats,
        setStats] =
        useState({
            Crack: 0,
            Spall: 0,
            Corrosion: 0,
            Efflorescence: 0,
        });

    const handleFile = (e) => {

        const selected =
            e.target.files[0];

        if (!selected) return;

        setFile(selected);

        // Clear previous prediction

        setOverlayImage(null);

        setPredictionMask(null);

        setStats({
            Crack: 0,
            Spall: 0,
            Corrosion: 0,
            Efflorescence: 0,
        });

        setOriginalImage(
            URL.createObjectURL(
                selected
            )
        );
    };

    const analyze = async () => {

        if (!file) return;

        setLoading(true);

        try {

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );

            const response =
                await api.post(
                    "/predict",
                    formData
                );

            setStats({
                Crack:
                    response.data.crack || 0,

                Spall:
                    response.data.spall || 0,

                Corrosion:
                    response.data.corrosion || 0,

                Efflorescence:
                    response.data.efflorescence || 0,
            });

            setOverlayImage(
                response.data.overlay_image
            );

            setPredictionMask(
                response.data.prediction_mask
            );

        } catch (err) {

            console.error(err);

        } finally {

            setLoading(false);

        }
    };

    const StatCard = ({
        label,
        value
    }) => (

        <Card
            sx={{
                p: 2,
                height: "100%",
                background:
                    "rgba(15,23,42,0.9)",
                border:
                    "1px solid rgba(255,255,255,0.08)",
            }}
        >
            <Typography
                variant="body2"
                color="gray"
            >
                {label}
            </Typography>

            <Typography
                variant="h4"
                mt={1}
            >
                {value}%
            </Typography>

            <LinearProgress
                variant="determinate"
                value={value}
                sx={{
                    mt: 2,
                    height: 8,
                    borderRadius: 10,
                }}
            />
        </Card>
    );

    const ImageCard = ({
        title,
        image,
        placeholder,
    }) => (

            <Card
                sx={{
                    p: 2,
                    width: "100%",
                    background: "#0f172a",
                    border:
                        "1px solid rgba(255,255,255,0.08)",

                    display: "flex",
                    flexDirection: "column",

                    minHeight: 520,
                }}
            >

            <Typography
                variant="h6"
                mb={2}
            >
                {title}
            </Typography>

            {image ? (
                <Box
                    sx={{
                        width: "100%",
                        height: 420,

                        overflow: "hidden",

                        borderRadius: 3,

                        background: "#020617",

                        border:
                            "1px dashed #334155",

                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >

                    <img
                        src={image}
                        alt={title}
                        style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "contain",
                            display: "block",
                        }}
                    />

                </Box>

            ) : (

                <Box
                    sx={{
                        flex: 1,
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        color: "#64748b",
                        border:
                            "1px dashed #334155",
                        borderRadius: 2,
                    }}
                >
                    <Typography
                        color="#64748b"
                    >
                        {loading
                            ? "Running segmentation..."
                            : placeholder}
                    </Typography>
                </Box>

            )}

        </Card>
    );

    const SummaryCard = () => {

        const detected = [];

        if (stats.Crack > 0) detected.push("Crack");
        if (stats.Corrosion > 0) detected.push("Corrosion");
        if (stats.Spall > 0) detected.push("Spall");
        if (stats.Efflorescence > 0) detected.push("Efflorescence");

        return (
            <Card
                sx={{
                    p: 3,
                    background: "#0f172a",
                    border: "1px solid rgba(255,255,255,0.08)",
                    height: "100%",
                }}
            >
                <Typography variant="h6" mb={2}>
                    Detected Defects
                </Typography>

                {detected.length > 0 ? (
                    <Stack spacing={1}>
                        {detected.map((item) => (
                            <Typography
                                key={item}
                                sx={{
                                    fontSize: 18,
                                    fontWeight: 600,
                                }}
                            >
                                ✓ {item}
                            </Typography>
                        ))}
                    </Stack>
                ) : (
                    <Typography color="#94a3b8">
                        No defects detected
                    </Typography>
                )}
            </Card>
        );
    };

    const LegendCard = () => (
        <Card
            sx={{
                p: 3,
                background: "#0f172a",
                border: "1px solid rgba(255,255,255,0.08)",
                height: "100%",
            }}
        >
            <Typography variant="h6" mb={2}>
                Legend
            </Typography>

            <Stack spacing={2}>

                <Stack direction="row" spacing={2} alignItems="center">
                    <Box
                        sx={{
                            width: 18,
                            height: 18,
                            bgcolor: "#0000ff",
                            borderRadius: "50%",
                        }}
                    />
                    <Typography>Crack</Typography>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                    <Box
                        sx={{
                            width: 18,
                            height: 18,
                            bgcolor: "#00ff00",
                            borderRadius: "50%",
                        }}
                    />
                    <Typography>Efflorescence</Typography>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                    <Box
                        sx={{
                            width: 18,
                            height: 18,
                            bgcolor: "#ff0000",
                            borderRadius: "50%",
                        }}
                    />
                    <Typography>Exposed Rebar / Corrosion</Typography>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                    <Box
                        sx={{
                            width: 18,
                            height: 18,
                            bgcolor: "#00ffff",
                            borderRadius: "50%",
                        }}
                    />
                    <Typography>Spalling</Typography>
                </Stack>

            </Stack>
        </Card>
    );

    return (

        <Box
            sx={{
                minHeight: "100vh",
                background:
                    "#020617",
                py: 4,
            }}
        >

            <Container
                maxWidth="xl"
            >

                {/* HEADER */}

                <Card
                    sx={{
                        p: 4,
                        mb: 3,
                        background:
                            "rgba(15,23,42,0.95)",
                        border:
                            "1px solid rgba(255,255,255,0.08)",
                    }}
                >
                    <Stack
                        direction="row"
                        justifyContent="space-between"
                        alignItems="center"
                    >

                        <Box>

                            <Typography
                                variant="h3"
                                fontWeight={700}
                            >
                                Infrastructure Defect
                                Segmentation
                            </Typography>

                            <Typography
                                sx={{
                                    mt: 1,
                                    color:
                                        "#94a3b8",
                                }}
                            >
                                SegFormer-B0 |
                                Infrastructure
                                Inspection Demo
                            </Typography>

                        </Box>

                        <Chip
                            label="Online"
                            color="success"
                        />

                    </Stack>
                </Card>

                {/* ACTION BAR */}

                <Card
                    sx={{
                        p: 3,
                        mb: 3,
                        background:
                            "rgba(15,23,42,0.95)",
                    }}
                >

                    <Stack
                        direction="row"
                        spacing={2}
                    >

                        <Button
                            variant="contained"
                            startIcon={
                                <CloudUploadIcon />
                            }
                            component="label"
                        >

                            Upload Image

                            <input
                                hidden
                                type="file"
                                accept=".jpg,.jpeg,.png"
                                onChange={
                                    handleFile
                                }
                            />

                        </Button>

                        <Button
                            variant="outlined"
                            startIcon={
                                <AnalyticsIcon />
                            }
                            onClick={analyze}
                            disabled={
                                !file || loading
                            }
                        >
                            {
                                loading
                                    ? "Analyzing..."
                                    : "Run Analysis"
                            }
                        </Button>

                    </Stack>

                </Card>

                {/* IMAGES */}

                <Grid
                    container
                    spacing={3}
                    sx={{
                        width: "100%",
                        m: 0,
                    }}
                >

                    <Grid
                        size={{
                            xs: 12,
                            md: 4,
                        }}
                        sx={{
                            display: "flex",
                        }}
                    >
                        <Box sx={{ width: "100%" }}>
                            <ImageCard
                                title="Original Image"
                                image={originalImage}
                                placeholder="Upload image to begin"
                            />
                        </Box>
                    </Grid>

                    <Grid
                        size={{
                            xs: 12,
                            md: 4,
                        }}
                        sx={{
                            display: "flex",
                        }}
                    >
                        <Box sx={{ width: "100%" }}>
                            <ImageCard
                                title="Segmentation Overlay"
                                image={overlayImage}
                                placeholder="Awaiting prediction"
                            />
                        </Box>
                    </Grid>

                    <Grid
                        size={{
                            xs: 12,
                            md: 4,
                        }}
                        sx={{
                            display: "flex",
                        }}
                    >
                        <Box sx={{ width: "100%" }}>
                            <ImageCard
                                title="Prediction Mask"
                                image={predictionMask}
                                placeholder="Awaiting prediction"
                            />
                        </Box>
                        
                    </Grid>

                </Grid>

                <Grid
                    container
                    spacing={3}
                    sx={{
                        width: "100%",
                        mt: 3,
                    }}
                >
                    <Grid
                        size={{
                            xs: 12,
                            md: 6,
                        }}
                    >
                        <SummaryCard />
                    </Grid>

                    <Grid
                        size={{
                            xs: 12,
                            md: 6,
                        }}
                    >
                        <LegendCard />

                    </Grid>

                </Grid>

            </Container>

        </Box>

    );
}