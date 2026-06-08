import {
    Card,
    Button,
    Stack
} from "@mui/material";

export default function UploadCard({
    onFileChange,
    onAnalyze
}) {

    return (
        <Card sx={{ p: 3 }}>

            <Stack
                direction="row"
                spacing={2}
            >

                <Button
                    variant="contained"
                    component="label"
                >
                    Upload Image

                    <input
                        hidden
                        type="file"
                        accept=".jpg,.jpeg,.png"
                        onChange={onFileChange}
                    />
                </Button>

                <Button
                    variant="outlined"
                    onClick={onAnalyze}
                >
                    Run Analysis
                </Button>

            </Stack>

        </Card>
    );
}