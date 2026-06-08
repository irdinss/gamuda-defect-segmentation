import {
    Card,
    Typography
} from "@mui/material";

export default function ImagePanel({
    title,
    image
}) {

    return (
        <Card
            sx={{
                p: 2,
                minHeight: 450,
            }}
        >
            <Typography mb={2}>
                {title}
            </Typography>

            {image && (
                <img
                    src={image}
                    alt={title}
                    style={{
                        width: "100%",
                        borderRadius: 12,
                    }}
                />
            )}
        </Card>
    );
}